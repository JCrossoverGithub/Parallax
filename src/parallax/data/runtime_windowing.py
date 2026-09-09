"""Incremental observation-window construction for replayed packet metadata."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt

from parallax.data.flows import FlowPacketAssignment
from parallax.data.vnat import CaptureMetadata, parse_capture_filename
from parallax.data.windowing import (
    ConnectionKey,
    ObservationWindow,
    VnatWindowError,
    WindowExtractionConfig,
    make_flow_id,
)


class RuntimeWindowError(VnatWindowError):
    """Raised when incremental packet assignments cannot form runtime windows."""


@dataclass(slots=True)
class _WindowBuffer:
    window_index: int
    timestamps: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)


class IncrementalWindowTracker:
    """Incrementally emit completed eligible observation windows."""

    __slots__ = (
        "_buffers",
        "_capture",
        "_capture_origin",
        "_config",
        "_current_window_index",
        "_finished",
        "_packet_number",
    )

    def __init__(
        self,
        capture_name: str,
        *,
        config: WindowExtractionConfig | None = None,
    ) -> None:
        self._capture: CaptureMetadata = parse_capture_filename(capture_name)
        self._config = config if config is not None else WindowExtractionConfig()
        self._capture_origin: float | None = None
        self._current_window_index: int | None = None
        self._buffers: dict[ConnectionKey, _WindowBuffer] = {}
        self._packet_number = 0
        self._finished = False

    def push(
        self,
        assignment: FlowPacketAssignment,
    ) -> tuple[ObservationWindow, ...]:
        """Consume one flow-assigned packet and emit newly completed windows."""
        if self._finished:
            raise RuntimeWindowError("cannot push packets after runtime windowing is finished")

        expected_packet_number = self._packet_number + 1
        if assignment.packet_number != expected_packet_number:
            raise RuntimeWindowError(
                f"expected packet {expected_packet_number}, got packet {assignment.packet_number}"
            )

        packet = assignment.packet
        if self._capture_origin is None:
            self._capture_origin = packet.timestamp_seconds

        capture_relative_timestamp = np.float64(packet.timestamp_seconds) - np.float64(
            self._capture_origin
        )
        window_index = int(np.floor(capture_relative_timestamp / self._config.window_seconds))

        current_window_index = self._current_window_index
        if current_window_index is not None and window_index < current_window_index:
            raise RuntimeWindowError(
                f"packet {assignment.packet_number}: window index precedes "
                "the current capture window"
            )

        completed: tuple[ObservationWindow, ...] = ()

        if current_window_index is None:
            self._current_window_index = window_index
        elif window_index > current_window_index:
            completed = self._flush_all()
            self._current_window_index = window_index

        buffer = self._buffers.get(assignment.connection)
        if buffer is None:
            buffer = _WindowBuffer(window_index=window_index)
            self._buffers[assignment.connection] = buffer

        buffer.timestamps.append(packet.timestamp_seconds)
        buffer.sizes.append(packet.size)
        buffer.directions.append(assignment.direction)

        self._packet_number = assignment.packet_number
        return completed

    def finish(self) -> tuple[ObservationWindow, ...]:
        """Flush final eligible windows and permanently close the tracker."""
        if self._finished:
            return ()

        self._finished = True
        windows = self._flush_all()
        self._buffers.clear()
        return windows

    def _flush_all(self) -> tuple[ObservationWindow, ...]:
        completed: list[ObservationWindow] = []

        for connection in sorted(tuple(self._buffers)):
            buffer = self._buffers.pop(connection)
            window = self._freeze_window(connection, buffer)

            if window is not None:
                completed.append(window)

        return tuple(completed)

    def _freeze_window(
        self,
        connection: ConnectionKey,
        buffer: _WindowBuffer,
    ) -> ObservationWindow | None:
        if not self._config.retains(len(buffer.timestamps)):
            return None

        capture_origin = cast(float, self._capture_origin)

        absolute_timestamps = np.asarray(buffer.timestamps, dtype=np.float64)
        capture_relative_timestamps = absolute_timestamps - capture_origin
        window_start = buffer.window_index * self._config.window_seconds

        flow_id = make_flow_id(self._capture.capture_id, connection)

        return ObservationWindow(
            window_id=f"{self._capture.capture_id}:{flow_id}:{buffer.window_index}",
            capture=self._capture,
            flow_id=flow_id,
            connection=connection,
            window_index=buffer.window_index,
            start_offset_seconds=window_start,
            end_offset_seconds=(buffer.window_index + 1) * self._config.window_seconds,
            timestamps=_readonly(
                capture_relative_timestamps - window_start,
                np.float64,
            ),
            sizes=_readonly(buffer.sizes, np.int64),
            directions=_readonly(buffer.directions, np.int8),
        )


def _readonly[ScalarType: np.generic](
    values: npt.ArrayLike,
    dtype: type[ScalarType],
) -> npt.NDArray[ScalarType]:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
