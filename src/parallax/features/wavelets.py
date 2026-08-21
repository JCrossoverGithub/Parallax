"""Typed stationary-wavelet transforms and VNAT wavelet feature calculations."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

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
Float32Array = npt.NDArray[np.float32]
WAVELET_LOG_EPSILON: Final = 1e-4


class WaveletNormalization(StrEnum):
    """Explicit PyWavelets normalization candidates for release comparison."""

    PYWT_DEFAULT = "pywt-default"
    ENERGY_PRESERVING = "energy-preserving"


@dataclass(frozen=True)
class DirectionalWaveletFeatures:
    """Four feature families calculated for one packet direction."""

    relative_energy: Float32Array
    shannon_entropy: Float32Array
    log_mean_absolute: Float32Array
    log_standard_deviation: Float32Array

    def __post_init__(self) -> None:
        for values in (
            self.relative_energy,
            self.shannon_entropy,
            self.log_mean_absolute,
            self.log_standard_deviation,
        ):
            if values.shape != (WAVELET_BAND_COUNT,):
                raise FeatureContractError("invalid directional wavelet feature shape")
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise FeatureContractError("invalid directional wavelet feature values")
            values.setflags(write=False)


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


def calculate_directional_wavelet_features(
    signal: npt.ArrayLike,
) -> DirectionalWaveletFeatures:
    """Calculate the four VNAT wavelet feature families for one direction."""
    bands = stationary_haar_bands(
        signal,
        normalization=WaveletNormalization.ENERGY_PRESERVING,
    )
    coefficients = np.stack(bands)
    squared = np.square(coefficients)
    band_energies = np.sum(squared, axis=1)
    total_energy = float(np.sum(band_energies))

    if total_energy == 0.0:
        relative_energy = np.zeros(WAVELET_BAND_COUNT, dtype=np.float64)
    else:
        relative_energy = band_energies / total_energy

    shannon_entropy = np.zeros(WAVELET_BAND_COUNT, dtype=np.float64)
    for band_index, band_energy in enumerate(band_energies):
        if band_energy == 0.0:
            continue
        probabilities = squared[band_index] / band_energy
        nonzero = probabilities > 0.0
        shannon_entropy[band_index] = -np.sum(
            probabilities[nonzero] * np.log2(probabilities[nonzero])
        )

    log_mean_absolute = np.log(np.mean(np.abs(coefficients), axis=1) + WAVELET_LOG_EPSILON)
    log_standard_deviation = np.log(np.std(coefficients, axis=1) + WAVELET_LOG_EPSILON)

    return DirectionalWaveletFeatures(
        relative_energy=_readonly_float32(relative_energy),
        shannon_entropy=_readonly_float32(shannon_entropy),
        log_mean_absolute=_readonly_float32(log_mean_absolute),
        log_standard_deviation=_readonly_float32(log_standard_deviation),
    )


def calculate_wavelet_feature_vector(
    incoming_signal: npt.ArrayLike,
    outgoing_signal: npt.ArrayLike,
) -> Float32Array:
    """Return the 104 release-compatible wavelet features in schema order."""
    incoming = calculate_directional_wavelet_features(incoming_signal)
    outgoing = calculate_directional_wavelet_features(outgoing_signal)
    values = np.concatenate(
        (
            incoming.relative_energy,
            outgoing.relative_energy,
            incoming.shannon_entropy,
            outgoing.shannon_entropy,
            incoming.log_mean_absolute,
            outgoing.log_mean_absolute,
            incoming.log_standard_deviation,
            outgoing.log_standard_deviation,
        )
    ).astype(np.float32, copy=False)
    values.setflags(write=False)
    return values


def _readonly_float32(values: npt.ArrayLike) -> Float32Array:
    normalized = np.array(values, dtype=np.float32, copy=True)
    normalized.setflags(write=False)
    return normalized
