"""Exact schema and numerical invariants for VNAT release 1 features."""

from typing import Final

import numpy as np
import numpy.typing as npt

from parallax.data.vnat import FEATURE_COUNT

Float32Array = npt.NDArray[np.float32]

FEATURE_SCHEMA_VERSION: Final = "vnat-feature-1"
WINDOW_SAMPLE_COUNT: Final = 4096
WAVELET_NAME: Final = "haar"
STATIONARY_WAVELET_LEVEL: Final = 12
WAVELET_BAND_COUNT: Final = STATIONARY_WAVELET_LEVEL + 1

FLOW_STATISTIC_COLUMNS: Final = (
    "out_iat_min",
    "out_iat_max",
    "out_iat_mean",
    "out_iat_std_dev",
    "in_iat_min",
    "in_iat_max",
    "in_iat_mean",
    "in_iat_std_dev",
    "flow_iat_min",
    "flow_iat_max",
    "flow_iat_mean",
    "flow_iat_std_dev",
    "active_min",
    "active_max",
    "active_mean",
    "active_std_dev",
    "idle_min",
    "idle_max",
    "idle_mean",
    "idle_std_dev",
    "log_bytes_per_sec",
    "log_total_outgoing_packets",
    "log_total_incoming_packets",
    "log_total_outgoing_bytes",
    "log_total_incoming_bytes",
)

_WAVELET_COLUMN_PREFIXES: Final = (
    "in_rel_eng",
    "out_rel_eng",
    "in_shannon_entropy",
    "out_shannon_entropy",
    "in_log_mean_abs_detail_coeffs",
    "out_log_mean_abs_detail_coeffs",
    "in_log_std_dev_detail_coeffs",
    "out_log_std_dev_detail_coeffs",
)
WAVELET_FEATURE_COLUMNS: Final = tuple(
    f"{prefix}_{band}" for prefix in _WAVELET_COLUMN_PREFIXES for band in range(WAVELET_BAND_COUNT)
)
FEATURE_COLUMNS: Final = FLOW_STATISTIC_COLUMNS + WAVELET_FEATURE_COLUMNS


class FeatureContractError(ValueError):
    """Raised when feature values violate the versioned numerical contract."""


def normalize_feature_vector(values: npt.ArrayLike) -> Float32Array:
    """Return an owned, finite, read-only float32 vector in schema order."""
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureContractError("feature values must be numeric") from error

    if vector.shape != (FEATURE_COUNT,):
        raise FeatureContractError(
            f"expected {FEATURE_COUNT} feature values, got shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise FeatureContractError("feature values must all be finite")

    normalized = np.array(vector, dtype=np.float32, copy=True)
    normalized.setflags(write=False)
    return normalized
