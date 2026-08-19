"""Deterministic observation-window extraction for VNAT release 1."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from parallax.data.vnat import (
    MIN_PACKETS_PER_WINDOW,
    RAW_COLUMNS,
    WINDOW_SECONDS,
    CaptureMetadata,
    parse_capture_filename,
)

ConnectionKey = tuple[str, int, str, int, int]
FloatArray = npt.NDArray[np.float64]
IntegerArray = npt.NDArray[np.int64]
DirectionArray = npt.NDArray[np.int8]
FLOW_ID_HEX_LENGTH: Final = 24


class WindowThresholdPolicy(StrEnum):
    """Supported interpretations of the published packet threshold."""

    RELEASE_COMPATIBLE = "release-compatible"
    PAPER_LITERAL = "paper-literal"


@dataclass(frozen=True, slots=True)
class WindowExtractionConfig:
    """Versioned parameters controlling observation-window eligibility."""

    window_seconds: float = WINDOW_SECONDS
    minimum_packets: int = MIN_PACKETS_PER_WINDOW
    threshold_policy: WindowThresholdPolicy = WindowThresholdPolicy.RELEASE_COMPATIBLE

    def __post_init__(self) -> None:
        if not isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and greater than zero")
        if self.minimum_packets < 1:
            raise ValueError("minimum_packets must be at least one")
        if not isinstance(self.threshold_policy, WindowThresholdPolicy):
            raise TypeError("threshold_policy must be a WindowThresholdPolicy")

    def retains(self, packet_count: int) -> bool:
        """Return whether a window satisfies the selected threshold semantics."""
        if self.threshold_policy is WindowThresholdPolicy.RELEASE_COMPATIBLE:
            return packet_count > self.minimum_packets
        return packet_count >= self.minimum_packets


@dataclass(frozen=True, slots=True, eq=False)
class ObservationWindow:
    """One eligible connection window with capture-relative packet metadata."""

    window_id: str
    capture: CaptureMetadata
    flow_id: str
    connection: ConnectionKey
    window_index: int
    start_offset_seconds: float
    end_offset_seconds: float
    timestamps: FloatArray
    sizes: IntegerArray
    directions: DirectionArray

    @property
    def packet_count(self) -> int:
        """Number of packets represented by this window."""
        return int(self.timestamps.size)


class VnatWindowError(ValueError):
    """Raised when raw VNAT records cannot be windowed deterministically."""


def extract_capture_windows(
    frame: pd.DataFrame,
    *,
    config: WindowExtractionConfig | None = None,
) -> Iterator[ObservationWindow]:
    """Validate and lazily extract eligible windows from exactly one capture."""
    selected_config = config or WindowExtractionConfig()
    capture, capture_origin = _validate_capture_frame(frame)
    return _iter_capture_windows(frame, capture, capture_origin, selected_config)


def _validate_capture_frame(frame: pd.DataFrame) -> tuple[CaptureMetadata, float]:
    if tuple(frame.columns) != RAW_COLUMNS:
        raise VnatWindowError(f"expected raw columns {RAW_COLUMNS!r}, got {tuple(frame.columns)!r}")
    if frame.empty:
        raise VnatWindowError("cannot extract windows from an empty capture")

    file_names = frame["file_names"].unique().tolist()
    if len(file_names) != 1 or not isinstance(file_names[0], str):
        raise VnatWindowError("expected records from exactly one capture filename")

    capture = parse_capture_filename(file_names[0])
    capture_origin = float("inf")
    connections: set[ConnectionKey] = set()

    for row_number, row in enumerate(frame.itertuples(index=False, name=None)):
        connection, timestamps, sizes, directions, _ = row
        normalized_connection = _normalize_connection(connection, row_number)
        if normalized_connection in connections:
            raise VnatWindowError(
                f"row {row_number}: duplicate connection {normalized_connection!r}"
            )
        connections.add(normalized_connection)

        timestamp_array, _, _ = _validate_packet_arrays(
            timestamps,
            sizes,
            directions,
            row_number,
        )
        capture_origin = min(capture_origin, float(timestamp_array.min()))

    return capture, capture_origin


def _iter_capture_windows(
    frame: pd.DataFrame,
    capture: CaptureMetadata,
    capture_origin: float,
    config: WindowExtractionConfig,
) -> Iterator[ObservationWindow]:
    rows = [
        (row_number, row, _normalize_connection(row[0], row_number))
        for row_number, row in enumerate(frame.itertuples(index=False, name=None))
    ]
    rows.sort(key=lambda item: item[2])

    for row_number, row, normalized_connection in rows:
        _, timestamps, sizes, directions, _ = row
        timestamp_array, size_array, direction_array = _validate_packet_arrays(
            timestamps,
            sizes,
            directions,
            row_number,
        )

        order = np.argsort(timestamp_array, kind="stable")
        ordered_timestamps = timestamp_array[order]
        ordered_sizes = size_array[order]
        ordered_directions = direction_array[order]
        capture_relative_timestamps = ordered_timestamps - capture_origin
        window_indices = np.floor(capture_relative_timestamps / config.window_seconds).astype(
            np.int64
        )

        flow_id = _make_flow_id(capture.capture_id, normalized_connection)
        boundaries = (np.flatnonzero(np.diff(window_indices)) + 1).tolist()
        start = 0

        for end in [*boundaries, int(window_indices.size)]:
            packet_count = end - start
            window_index = int(window_indices[start])

            if config.retains(packet_count):
                yield ObservationWindow(
                    window_id=f"{capture.capture_id}:{flow_id}:{window_index}",
                    capture=capture,
                    flow_id=flow_id,
                    connection=normalized_connection,
                    window_index=window_index,
                    start_offset_seconds=window_index * config.window_seconds,
                    end_offset_seconds=(window_index + 1) * config.window_seconds,
                    timestamps=_readonly(
                        capture_relative_timestamps[start:end]
                        - window_index * config.window_seconds,
                        np.float64,
                    ),
                    sizes=_readonly(ordered_sizes[start:end], np.int64),
                    directions=_readonly(ordered_directions[start:end], np.int8),
                )

            start = end


def _normalize_connection(value: object, row_number: int) -> ConnectionKey:
    if not isinstance(value, tuple) or len(value) != 5:
        raise VnatWindowError(f"row {row_number}: connection must be a five-tuple")

    source, source_port, destination, destination_port, protocol = value
    if not isinstance(source, str) or not isinstance(destination, str):
        raise VnatWindowError(f"row {row_number}: connection addresses must be strings")

    numeric_values = (source_port, destination_port, protocol)
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in numeric_values):
        raise VnatWindowError(f"row {row_number}: connection ports and protocol must be integers")

    normalized = (
        source,
        int(source_port),
        destination,
        int(destination_port),
        int(protocol),
    )
    if not 0 <= normalized[1] <= 65_535 or not 0 <= normalized[3] <= 65_535:
        raise VnatWindowError(f"row {row_number}: connection port is out of range")
    if not 0 <= normalized[4] <= 255:
        raise VnatWindowError(f"row {row_number}: connection protocol is out of range")
    return normalized


def _validate_packet_arrays(
    timestamps: object,
    sizes: object,
    directions: object,
    row_number: int,
) -> tuple[FloatArray, IntegerArray, DirectionArray]:
    try:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VnatWindowError(f"row {row_number}: timestamps must be numeric") from error

    size_array = np.asarray(sizes)
    direction_array = np.asarray(directions)
    arrays = (timestamp_array, size_array, direction_array)

    if any(array.ndim != 1 for array in arrays):
        raise VnatWindowError(f"row {row_number}: packet arrays must be one-dimensional")
    if timestamp_array.size == 0:
        raise VnatWindowError(f"row {row_number}: packet arrays cannot be empty")
    if not all(array.size == timestamp_array.size for array in arrays[1:]):
        raise VnatWindowError(f"row {row_number}: packet-array lengths do not match")
    if not np.isfinite(timestamp_array).all():
        raise VnatWindowError(f"row {row_number}: timestamps must be finite")
    if not np.issubdtype(size_array.dtype, np.integer):
        raise VnatWindowError(f"row {row_number}: packet sizes must be integers")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise VnatWindowError(f"row {row_number}: packet directions must be integers")

    normalized_sizes = size_array.astype(np.int64, copy=False)
    integer_directions = direction_array.astype(np.int64, copy=False)
    if np.any(normalized_sizes < 0):
        raise VnatWindowError(f"row {row_number}: packet sizes cannot be negative")
    if not np.isin(integer_directions, (0, 1)).all():
        raise VnatWindowError(f"row {row_number}: packet directions must be zero or one")

    return timestamp_array, normalized_sizes, integer_directions.astype(np.int8, copy=False)


def _make_flow_id(capture_id: str, connection: ConnectionKey) -> str:
    encoded = "\0".join((capture_id, *(str(item) for item in connection))).encode()
    return sha256(encoded).hexdigest()[:FLOW_ID_HEX_LENGTH]


def _readonly[ScalarType: np.generic](
    values: npt.ArrayLike,
    dtype: type[ScalarType],
) -> npt.NDArray[ScalarType]:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
