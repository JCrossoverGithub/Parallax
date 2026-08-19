"""Release-compatible aggregate flow-statistic calculations."""

import numpy as np
import numpy.typing as npt

from parallax.features.schema import FeatureContractError

Float32Array = npt.NDArray[np.float32]


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


def _interarrival_statistics(timestamps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    differences = np.diff(timestamps)
    if differences.size == 0:
        return np.zeros(4, dtype=np.float64)

    return np.asarray(
        (
            np.min(differences),
            np.max(differences),
            np.mean(differences),
            np.std(differences),
        ),
        dtype=np.float64,
    )


def _normalize_packet_timing(
    timestamps: npt.ArrayLike,
    directions: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    try:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureContractError("packet timestamps must be numeric") from error

    direction_array = np.asarray(directions)
    if timestamp_array.ndim != 1 or direction_array.ndim != 1:
        raise FeatureContractError("packet timing arrays must be one-dimensional")
    if timestamp_array.size == 0:
        raise FeatureContractError("packet timing arrays cannot be empty")
    if timestamp_array.size != direction_array.size:
        raise FeatureContractError("packet timing array lengths do not match")
    if not np.isfinite(timestamp_array).all():
        raise FeatureContractError("packet timestamps must all be finite")
    if np.any(np.diff(timestamp_array) < 0.0):
        raise FeatureContractError("packet timestamps must be nondecreasing")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise FeatureContractError("packet directions must be integers")

    integer_directions = direction_array.astype(np.int64, copy=False)
    if not np.isin(integer_directions, (0, 1)).all():
        raise FeatureContractError("packet directions must be zero or one")

    return timestamp_array, integer_directions.astype(np.int8, copy=False)
