"""Release-compatible aggregate flow-statistic calculations."""

from typing import Final

import numpy as np
import numpy.typing as npt

from parallax.features.schema import FeatureContractError

Float32Array = npt.NDArray[np.float32]
ACTIVITY_TIMEOUT_SECONDS: Final = 5.0


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
    direction_array = np.asarray(directions)

    if direction_array.ndim != 1:
        raise FeatureContractError("packet timing arrays must be one-dimensional")
    if timestamp_array.size != direction_array.size:
        raise FeatureContractError("packet timing array lengths do not match")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise FeatureContractError("packet directions must be integers")

    integer_directions = direction_array.astype(np.int64, copy=False)
    if not np.isin(integer_directions, (0, 1)).all():
        raise FeatureContractError("packet directions must be zero or one")

    return timestamp_array, integer_directions.astype(np.int8, copy=False)


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
