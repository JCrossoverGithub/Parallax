import numpy as np
import pytest

from parallax.features import (
    FeatureContractError,
    WaveletNormalization,
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
