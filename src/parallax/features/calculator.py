"""End-to-end construction of versioned VNAT feature vectors."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt

from parallax.data.vnat import TIME_BIN_SECONDS, WINDOW_SECONDS
from parallax.features.flow_statistics import (
    ByteTotalPolicy,
    calculate_active_idle_feature_vector,
    calculate_aggregate_feature_vector,
    calculate_interarrival_feature_vector,
)
from parallax.features.schema import (
    WINDOW_SAMPLE_COUNT,
    FeatureContractError,
    normalize_feature_vector,
)
from parallax.features.wavelets import calculate_wavelet_feature_vector

Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureCalculationConfig:
    """Versioned choices used to calculate one VNAT feature vector."""

    window_seconds: float = WINDOW_SECONDS
    time_bin_seconds: float = TIME_BIN_SECONDS
    byte_total_policy: ByteTotalPolicy = ByteTotalPolicy.RELEASE_COMPATIBLE

    def __post_init__(self) -> None:
        if not isfinite(self.window_seconds) or self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be finite and greater than zero")
        if not isfinite(self.time_bin_seconds) or self.time_bin_seconds <= 0.0:
            raise ValueError("time_bin_seconds must be finite and greater than zero")
        sample_count = self.window_seconds / self.time_bin_seconds
        if not np.isclose(sample_count, WINDOW_SAMPLE_COUNT, rtol=0.0, atol=1e-9):
            raise ValueError(f"window and bin durations must produce {WINDOW_SAMPLE_COUNT} samples")
        if not isinstance(self.byte_total_policy, ByteTotalPolicy):
            raise TypeError("byte_total_policy must be a ByteTotalPolicy")


def build_directional_size_signals(
    timestamps: npt.ArrayLike,
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
    *,
    config: FeatureCalculationConfig | None = None,
) -> tuple[Float64Array, Float64Array]:
    """Aggregate packet sizes into incoming and outgoing time-bin signals."""
    selected_config = config or FeatureCalculationConfig()
    timestamp_array, size_array, direction_array = _normalize_window_packets(
        timestamps,
        sizes,
        directions,
        selected_config,
    )
    bin_indices = np.floor(timestamp_array / selected_config.time_bin_seconds).astype(np.int64)

    incoming = np.zeros(WINDOW_SAMPLE_COUNT, dtype=np.float64)
    outgoing = np.zeros(WINDOW_SAMPLE_COUNT, dtype=np.float64)
    incoming_packets = direction_array == 0
    outgoing_packets = direction_array == 1
    np.add.at(incoming, bin_indices[incoming_packets], size_array[incoming_packets])
    np.add.at(outgoing, bin_indices[outgoing_packets], size_array[outgoing_packets])
    incoming.setflags(write=False)
    outgoing.setflags(write=False)
    return incoming, outgoing


def calculate_feature_vector(
    timestamps: npt.ArrayLike,
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
    *,
    config: FeatureCalculationConfig | None = None,
) -> Float32Array:
    """Calculate all 129 VNAT features in the versioned schema order."""
    selected_config = config or FeatureCalculationConfig()
    incoming_signal, outgoing_signal = build_directional_size_signals(
        timestamps,
        sizes,
        directions,
        config=selected_config,
    )
    values = np.concatenate(
        (
            calculate_interarrival_feature_vector(timestamps, directions),
            calculate_active_idle_feature_vector(timestamps),
            calculate_aggregate_feature_vector(
                sizes,
                directions,
                window_seconds=selected_config.window_seconds,
                byte_total_policy=selected_config.byte_total_policy,
            ),
            calculate_wavelet_feature_vector(incoming_signal, outgoing_signal),
        )
    )
    return normalize_feature_vector(values)


def _normalize_window_packets(
    timestamps: npt.ArrayLike,
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
    config: FeatureCalculationConfig,
) -> tuple[Float64Array, npt.NDArray[np.int64], npt.NDArray[np.int8]]:
    try:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureContractError("packet timestamps must be numeric") from error

    size_array = np.asarray(sizes)
    direction_array = np.asarray(directions)
    arrays = (timestamp_array, size_array, direction_array)
    if any(array.ndim != 1 for array in arrays):
        raise FeatureContractError("window packet arrays must be one-dimensional")
    if timestamp_array.size == 0:
        raise FeatureContractError("window packet arrays cannot be empty")
    if any(array.size != timestamp_array.size for array in arrays[1:]):
        raise FeatureContractError("window packet array lengths do not match")
    if not np.isfinite(timestamp_array).all():
        raise FeatureContractError("packet timestamps must all be finite")
    if np.any(np.diff(timestamp_array) < 0.0):
        raise FeatureContractError("packet timestamps must be nondecreasing")
    if np.any(timestamp_array < 0.0) or np.any(timestamp_array >= config.window_seconds):
        raise FeatureContractError("packet timestamps must fall within the observation window")
    if not np.issubdtype(size_array.dtype, np.integer):
        raise FeatureContractError("packet sizes must be integers")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise FeatureContractError("packet directions must be integers")

    normalized_sizes = size_array.astype(np.int64, copy=False)
    normalized_directions = direction_array.astype(np.int64, copy=False)
    if np.any(normalized_sizes < 0):
        raise FeatureContractError("packet sizes cannot be negative")
    if not np.isin(normalized_directions, (0, 1)).all():
        raise FeatureContractError("packet directions must be zero or one")

    return (
        timestamp_array,
        normalized_sizes,
        normalized_directions.astype(np.int8, copy=False),
    )
