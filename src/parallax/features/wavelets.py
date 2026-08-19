"""Typed stationary-wavelet boundary for the untyped PyWavelets package."""

from enum import StrEnum
from typing import cast

import numpy as np
import numpy.typing as npt
import pywt  # type: ignore[import-untyped]

from parallax.features.schema import (
    STATIONARY_WAVELET_LEVEL,
    WAVELET_BAND_COUNT,
    WAVELET_NAME,
    WINDOW_SAMPLE_COUNT,
    FeatureContractError,
)

Float64Array = npt.NDArray[np.float64]


class WaveletNormalization(StrEnum):
    """Explicit PyWavelets normalization candidates for release comparison."""

    PYWT_DEFAULT = "pywt-default"
    ENERGY_PRESERVING = "energy-preserving"


def stationary_haar_bands(
    signal: npt.ArrayLike,
    *,
    normalization: WaveletNormalization,
) -> tuple[Float64Array, ...]:
    """Return the deepest approximation followed by twelve SWT detail bands."""
    if not isinstance(normalization, WaveletNormalization):
        raise TypeError("normalization must be a WaveletNormalization")

    try:
        samples = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureContractError("wavelet signal values must be numeric") from error

    if samples.shape != (WINDOW_SAMPLE_COUNT,):
        raise FeatureContractError(
            f"expected {WINDOW_SAMPLE_COUNT} wavelet samples, got shape {samples.shape}"
        )
    if not np.isfinite(samples).all():
        raise FeatureContractError("wavelet signal values must all be finite")

    transformed = cast(
        "list[Float64Array]",
        pywt.swt(
            samples,
            WAVELET_NAME,
            level=STATIONARY_WAVELET_LEVEL,
            trim_approx=True,
            norm=normalization is WaveletNormalization.ENERGY_PRESERVING,
        ),
    )
    if len(transformed) != WAVELET_BAND_COUNT:
        raise FeatureContractError(
            f"expected {WAVELET_BAND_COUNT} wavelet bands, got {len(transformed)}"
        )

    bands: list[Float64Array] = []
    for band in transformed:
        normalized = np.array(band, dtype=np.float64, copy=True)
        if normalized.shape != (WINDOW_SAMPLE_COUNT,) or not np.isfinite(normalized).all():
            raise FeatureContractError("PyWavelets returned an invalid coefficient band")
        normalized.setflags(write=False)
        bands.append(normalized)

    return tuple(bands)
