import numpy as np
import pytest

from parallax.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FLOW_STATISTIC_COLUMNS,
    STATIONARY_WAVELET_LEVEL,
    WAVELET_BAND_COUNT,
    WAVELET_FEATURE_COLUMNS,
    WAVELET_NAME,
    WINDOW_SAMPLE_COUNT,
    FeatureContractError,
    normalize_feature_vector,
)


def test_feature_schema_matches_release_column_contract() -> None:
    assert FEATURE_SCHEMA_VERSION == "vnat-feature-1"
    assert WINDOW_SAMPLE_COUNT == 4096
    assert WAVELET_NAME == "haar"
    assert STATIONARY_WAVELET_LEVEL == 12
    assert WAVELET_BAND_COUNT == 13
    assert WINDOW_SAMPLE_COUNT * 0.01 == 40.96
    assert len(FLOW_STATISTIC_COLUMNS) == 25
    assert len(WAVELET_FEATURE_COLUMNS) == 104
    assert len(FEATURE_COLUMNS) == 129
    assert len(set(FEATURE_COLUMNS)) == 129
    assert FEATURE_COLUMNS[:4] == (
        "out_iat_min",
        "out_iat_max",
        "out_iat_mean",
        "out_iat_std_dev",
    )
    assert FEATURE_COLUMNS[20:25] == (
        "log_bytes_per_sec",
        "log_total_outgoing_packets",
        "log_total_incoming_packets",
        "log_total_outgoing_bytes",
        "log_total_incoming_bytes",
    )
    assert WAVELET_FEATURE_COLUMNS[:13] == tuple(f"in_rel_eng_{band}" for band in range(13))
    assert FEATURE_COLUMNS[-13:] == tuple(
        f"out_log_std_dev_detail_coeffs_{band}" for band in range(13)
    )


def test_normalizes_feature_vector_as_owned_read_only_float32() -> None:
    source = np.arange(129, dtype=np.float64)

    vector = normalize_feature_vector(source)
    source[0] = -1.0

    assert vector.dtype == np.float32
    assert vector.flags.owndata is True
    assert vector.flags.writeable is False
    assert vector[0] == 0.0


@pytest.mark.parametrize("length", [0, 128, 130])
def test_rejects_wrong_feature_vector_length(length: int) -> None:
    with pytest.raises(FeatureContractError, match="expected 129 feature values"):
        normalize_feature_vector(np.zeros(length))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_feature_values(value: float) -> None:
    values = np.zeros(129)
    values[10] = value

    with pytest.raises(FeatureContractError, match="must all be finite"):
        normalize_feature_vector(values)


def test_rejects_nonnumeric_feature_values() -> None:
    with pytest.raises(FeatureContractError, match="must be numeric"):
        normalize_feature_vector(["invalid"] * 129)
