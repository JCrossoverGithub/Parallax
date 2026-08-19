"""Release-compatible aggregate flow-statistic calculations."""

from enum import StrEnum
from math import isfinite
from typing import Final

import numpy as np
import numpy.typing as npt

from parallax.features.schema import FeatureContractError

Float32Array = npt.NDArray[np.float32]
ACTIVITY_TIMEOUT_SECONDS: Final = 5.0
AGGREGATE_LOG_EPSILON: Final = 1e-4


class ByteTotalPolicy(StrEnum):
    """Treatment of the duplicated byte-total fields in the released artifact."""

    RELEASE_COMPATIBLE = "release-compatible"
    CORRECTED = "corrected"


def calculate_interarrival_feature_vector(
    timestamps: npt.ArrayLike,
    directions: npt.ArrayLike,
) -> Float32Array:
    """Return outgoing, incoming, and whole-flow IAT statistics in schema order."""
    normalized_timestamps, normalized_directions = _normalize_packet_timing(
        timestamps,
        directions,
    )
    outgoing = _interarrival_statistics(normalized_timestamps[normalized_directions == 1])
    incoming = _interarrival_statistics(normalized_timestamps[normalized_directions == 0])
    flow = _interarrival_statistics(normalized_timestamps)
    values = np.concatenate((outgoing, incoming, flow)).astype(np.float32, copy=False)
    values.setflags(write=False)
    return values


def calculate_active_idle_feature_vector(
    timestamps: npt.ArrayLike,
) -> Float32Array:
    """Return active and idle duration statistics in release schema order."""
    normalized_timestamps = _normalize_timestamps(timestamps)
    gaps = np.diff(normalized_timestamps)
    split_indices = np.flatnonzero(gaps > ACTIVITY_TIMEOUT_SECONDS)

    active_starts = np.concatenate((np.asarray([0], dtype=np.int64), split_indices + 1))
    active_ends = np.concatenate(
        (
            split_indices,
            np.asarray([normalized_timestamps.size - 1], dtype=np.int64),
        )
    )
    active_durations = normalized_timestamps[active_ends] - normalized_timestamps[active_starts]
    active_durations = active_durations[active_durations > 0.0]
    idle_durations = gaps[split_indices] - ACTIVITY_TIMEOUT_SECONDS

    values = np.concatenate(
        (
            _summary_statistics(active_durations),
            _summary_statistics(idle_durations),
        )
    ).astype(np.float32, copy=False)
    values.setflags(write=False)
    return values


def calculate_aggregate_feature_vector(
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
    *,
    window_seconds: float,
    byte_total_policy: ByteTotalPolicy = ByteTotalPolicy.RELEASE_COMPATIBLE,
) -> Float32Array:
    """Return the five log-scaled aggregate features in release schema order."""
    if not isfinite(window_seconds) or window_seconds <= 0.0:
        raise FeatureContractError("window_seconds must be finite and greater than zero")
    if not isinstance(byte_total_policy, ByteTotalPolicy):
        raise TypeError("byte_total_policy must be a ByteTotalPolicy")

    normalized_sizes, normalized_directions = _normalize_packet_sizes(
        sizes,
        directions,
    )
    total_bytes = int(np.sum(normalized_sizes))
    if total_bytes <= 0:
        raise FeatureContractError("aggregate packet bytes must be greater than zero")

    outgoing = normalized_directions == 1
    incoming = normalized_directions == 0
    outgoing_count = int(np.count_nonzero(outgoing))
    incoming_count = int(np.count_nonzero(incoming))
    logged_outgoing_count = _logged_total(outgoing_count)
    logged_incoming_count = _logged_total(incoming_count)

    if byte_total_policy is ByteTotalPolicy.RELEASE_COMPATIBLE:
        logged_outgoing_bytes = logged_outgoing_count
        logged_incoming_bytes = logged_incoming_count
    else:
        logged_outgoing_bytes = _logged_total(int(np.sum(normalized_sizes[outgoing])))
        logged_incoming_bytes = _logged_total(int(np.sum(normalized_sizes[incoming])))

    values = np.asarray(
        (
            np.log(total_bytes / window_seconds),
            logged_outgoing_count,
            logged_incoming_count,
            logged_outgoing_bytes,
            logged_incoming_bytes,
        ),
        dtype=np.float32,
    )
    values.setflags(write=False)
    return values


def _logged_total(value: int) -> float:
    if value == 0:
        return 0.0
    return float(np.log(value + AGGREGATE_LOG_EPSILON))


def _interarrival_statistics(timestamps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return _summary_statistics(np.diff(timestamps))


def _summary_statistics(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if values.size == 0:
        return np.zeros(4, dtype=np.float64)

    return np.asarray(
        (
            np.min(values),
            np.max(values),
            np.mean(values),
            np.std(values),
        ),
        dtype=np.float64,
    )


def _normalize_packet_timing(
    timestamps: npt.ArrayLike,
    directions: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    timestamp_array = _normalize_timestamps(timestamps)
    direction_array = _normalize_directions(directions, timestamp_array.size)
    return timestamp_array, direction_array


def _normalize_packet_sizes(
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int8]]:
    size_array = np.asarray(sizes)
    if size_array.ndim != 1:
        raise FeatureContractError("packet metadata arrays must be one-dimensional")
    if size_array.size == 0:
        raise FeatureContractError("packet metadata arrays cannot be empty")
    if not np.issubdtype(size_array.dtype, np.integer):
        raise FeatureContractError("packet sizes must be integers")

    normalized_sizes = size_array.astype(np.int64, copy=False)
    if np.any(normalized_sizes < 0):
        raise FeatureContractError("packet sizes cannot be negative")

    direction_array = _normalize_directions(directions, size_array.size)
    return normalized_sizes, direction_array


def _normalize_directions(
    directions: npt.ArrayLike,
    expected_size: int,
) -> npt.NDArray[np.int8]:
    direction_array = np.asarray(directions)

    if direction_array.ndim != 1:
        raise FeatureContractError("packet direction array must be one-dimensional")
    if expected_size != direction_array.size:
        raise FeatureContractError("packet metadata array lengths do not match")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise FeatureContractError("packet directions must be integers")

    integer_directions = direction_array.astype(np.int64, copy=False)
    if not np.isin(integer_directions, (0, 1)).all():
        raise FeatureContractError("packet directions must be zero or one")

    return integer_directions.astype(np.int8, copy=False)


def _normalize_timestamps(timestamps: npt.ArrayLike) -> npt.NDArray[np.float64]:
    try:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureContractError("packet timestamps must be numeric") from error

    if timestamp_array.ndim != 1:
        raise FeatureContractError("packet timing arrays must be one-dimensional")
    if timestamp_array.size == 0:
        raise FeatureContractError("packet timing arrays cannot be empty")
    if not np.isfinite(timestamp_array).all():
        raise FeatureContractError("packet timestamps must all be finite")
    if np.any(np.diff(timestamp_array) < 0.0):
        raise FeatureContractError("packet timestamps must be nondecreasing")
    return timestamp_array
