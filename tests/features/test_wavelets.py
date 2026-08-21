import numpy as np
import pytest

from parallax.features import (
    WAVELET_LOG_EPSILON,
    DirectionalWaveletFeatures,
    FeatureContractError,
    WaveletNormalization,
    calculate_directional_wavelet_features,
    calculate_wavelet_feature_vector,
    stationary_haar_bands,
)


@pytest.mark.parametrize("normalization", list(WaveletNormalization))
def test_stationary_haar_bands_preserve_shape_and_are_read_only(
    normalization: WaveletNormalization,
) -> None:
    signal = np.ones(4096)

    bands = stationary_haar_bands(signal, normalization=normalization)

    assert len(bands) == 13
    assert all(band.shape == (4096,) for band in bands)
    assert all(band.dtype == np.float64 for band in bands)
    assert all(band.flags.writeable is False for band in bands)
    assert np.all(bands[0] > 0.0)
    assert all(np.allclose(band, 0.0) for band in bands[1:])


def test_energy_preserving_normalization_partitions_signal_energy() -> None:
    signal = np.zeros(4096)
    signal[123] = 4.0

    bands = stationary_haar_bands(
        signal,
        normalization=WaveletNormalization.ENERGY_PRESERVING,
    )

    input_energy = np.sum(np.square(signal))
    coefficient_energy = sum(float(np.sum(np.square(band))) for band in bands)
    assert coefficient_energy == pytest.approx(input_energy)


def test_pywavelets_default_and_energy_preserving_results_are_distinct() -> None:
    signal = np.zeros(4096)
    signal[123] = 1.0

    default = stationary_haar_bands(
        signal,
        normalization=WaveletNormalization.PYWT_DEFAULT,
    )
    normalized = stationary_haar_bands(
        signal,
        normalization=WaveletNormalization.ENERGY_PRESERVING,
    )

    assert any(
        not np.array_equal(left, right) for left, right in zip(default, normalized, strict=True)
    )


def test_rejects_wrong_number_of_pywavelets_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "parallax.features.wavelets.pywt.swt",
        lambda *args, **kwargs: [np.zeros(4096)] * 12,
    )

    with pytest.raises(FeatureContractError, match="expected 13 wavelet bands"):
        stationary_haar_bands(
            np.zeros(4096),
            normalization=WaveletNormalization.PYWT_DEFAULT,
        )


def test_rejects_invalid_pywavelets_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "parallax.features.wavelets.pywt.swt",
        lambda *args, **kwargs: [np.zeros(4096)] * 12 + [np.zeros(10)],
    )

    with pytest.raises(FeatureContractError, match="invalid coefficient band"):
        stationary_haar_bands(
            np.zeros(4096),
            normalization=WaveletNormalization.PYWT_DEFAULT,
        )


@pytest.mark.parametrize("length", [0, 4095, 4097])
def test_rejects_wrong_wavelet_signal_length(length: int) -> None:
    with pytest.raises(FeatureContractError, match="expected 4096 wavelet samples"):
        stationary_haar_bands(
            np.zeros(length),
            normalization=WaveletNormalization.PYWT_DEFAULT,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_wavelet_signal(value: float) -> None:
    signal = np.zeros(4096)
    signal[10] = value

    with pytest.raises(FeatureContractError, match="must all be finite"):
        stationary_haar_bands(
            signal,
            normalization=WaveletNormalization.PYWT_DEFAULT,
        )


def test_rejects_nonnumeric_wavelet_signal() -> None:
    with pytest.raises(FeatureContractError, match="must be numeric"):
        stationary_haar_bands(
            ["invalid"] * 4096,
            normalization=WaveletNormalization.PYWT_DEFAULT,
        )


def test_rejects_invalid_normalization() -> None:
    with pytest.raises(TypeError, match="must be a WaveletNormalization"):
        stationary_haar_bands(
            np.zeros(4096),
            normalization="pywt-default",  # type: ignore[arg-type]
        )


def test_directional_features_for_empty_signal_match_release_sentinels() -> None:
    features = calculate_directional_wavelet_features(np.zeros(4096))

    assert np.array_equal(features.relative_energy, np.zeros(13, dtype=np.float32))
    assert np.array_equal(features.shannon_entropy, np.zeros(13, dtype=np.float32))
    expected_log = np.float32(np.log(WAVELET_LOG_EPSILON))
    assert np.array_equal(features.log_mean_absolute, np.full(13, expected_log))
    assert np.array_equal(features.log_standard_deviation, np.full(13, expected_log))
    assert all(
        values.flags.writeable is False
        for values in (
            features.relative_energy,
            features.shannon_entropy,
            features.log_mean_absolute,
            features.log_standard_deviation,
        )
    )


def test_directional_features_follow_validated_formulas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = tuple(np.full(4096, band_index + 1.0, dtype=np.float64) for band_index in range(13))
    monkeypatch.setattr(
        "parallax.features.wavelets.stationary_haar_bands",
        lambda signal, *, normalization: bands,
    )

    features = calculate_directional_wavelet_features(np.zeros(4096))
    band_values = np.arange(1.0, 14.0)
    expected_energy = np.square(band_values)

    assert np.allclose(features.relative_energy, expected_energy / expected_energy.sum())
    assert np.array_equal(features.shannon_entropy, np.full(13, 12.0, dtype=np.float32))
    assert np.allclose(features.log_mean_absolute, np.log(band_values + 1e-4))
    assert np.array_equal(
        features.log_standard_deviation,
        np.full(13, np.float32(np.log(1e-4))),
    )


def test_combined_vector_uses_official_feature_family_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = DirectionalWaveletFeatures(
        relative_energy=np.full(13, 1, dtype=np.float32),
        shannon_entropy=np.full(13, 3, dtype=np.float32),
        log_mean_absolute=np.full(13, 5, dtype=np.float32),
        log_standard_deviation=np.full(13, 7, dtype=np.float32),
    )
    outgoing = DirectionalWaveletFeatures(
        relative_energy=np.full(13, 2, dtype=np.float32),
        shannon_entropy=np.full(13, 4, dtype=np.float32),
        log_mean_absolute=np.full(13, 6, dtype=np.float32),
        log_standard_deviation=np.full(13, 8, dtype=np.float32),
    )
    results = iter((incoming, outgoing))
    monkeypatch.setattr(
        "parallax.features.wavelets.calculate_directional_wavelet_features",
        lambda signal: next(results),
    )

    vector = calculate_wavelet_feature_vector(np.zeros(4096), np.zeros(4096))

    assert vector.shape == (104,)
    assert vector.dtype == np.float32
    assert vector.flags.writeable is False
    assert np.array_equal(vector, np.repeat(np.arange(1, 9, dtype=np.float32), 13))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((np.zeros(12, dtype=np.float32),) * 4, "shape"),
        (
            (
                np.full(13, np.nan, dtype=np.float32),
                np.zeros(13, dtype=np.float32),
                np.zeros(13, dtype=np.float32),
                np.zeros(13, dtype=np.float32),
            ),
            "values",
        ),
        (
            (
                np.zeros(13, dtype=np.float64),
                np.zeros(13, dtype=np.float32),
                np.zeros(13, dtype=np.float32),
                np.zeros(13, dtype=np.float32),
            ),
            "values",
        ),
    ],
)
def test_directional_feature_contract_rejects_invalid_arrays(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    message: str,
) -> None:
    with pytest.raises(FeatureContractError, match=message):
        DirectionalWaveletFeatures(*values)
