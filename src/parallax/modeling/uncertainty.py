"""Relative-Mahalanobis geometry and class-conditional OOD calibration."""

from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import ndtr
from torch import Tensor
from torch.nn import functional

from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.prototypes import EMBEDDING_DIMENSION

OOD_CALIBRATION_SCHEMA_VERSION: Final = "vnat-ood-calibration-1"
KDE_BANDWIDTH_METHOD: Final = "scott"

Float64Vector = npt.NDArray[np.float64]


class OODCalibrationError(ValueError):
    """Raised when uncertainty geometry or calibration violates its contract."""


@dataclass(frozen=True, slots=True)
class RelativeMahalanobisState:
    """Full-covariance support geometry used by the paper's OOD score."""

    class_means: Tensor
    class_covariances: Tensor
    global_mean: Tensor
    global_covariance: Tensor


@dataclass(frozen=True, slots=True)
class GaussianKDE1D:
    """Serializable one-dimensional Gaussian KDE with Scott bandwidth."""

    samples: Float64Vector
    bandwidth: float

    def __post_init__(self) -> None:
        if self.samples.ndim != 1 or self.samples.size < 2:
            raise OODCalibrationError("KDE samples must be a one-dimensional array of length two")
        if not np.isfinite(self.samples).all():
            raise OODCalibrationError("KDE samples must contain only finite values")
        if not isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise OODCalibrationError("KDE bandwidth must be finite and positive")

    def cdf(self, values: Float64Vector) -> Float64Vector:
        """Evaluate the fitted mixture CDF at each relative distance."""
        _validate_score_vector(values, name="KDE evaluation values")
        standardized = (values[:, None] - self.samples[None, :]) / self.bandwidth
        result = np.asarray(ndtr(standardized).mean(axis=1), dtype=np.float64)
        result.setflags(write=False)
        return result

    def as_dict(self) -> dict[str, object]:
        """Return the versioned, reproducible KDE configuration."""
        return {
            "bandwidth_method": KDE_BANDWIDTH_METHOD,
            "sample_count": int(self.samples.size),
            "bandwidth": self.bandwidth,
        }


def fit_relative_mahalanobis_state(
    support_embeddings: Tensor,
    support_labels: Tensor,
) -> RelativeMahalanobisState:
    """Fit class and global full-covariance geometry on balanced training support."""
    _validate_support(support_embeddings, support_labels)
    class_count = len(CATEGORY_LABELS)
    membership = functional.one_hot(support_labels, num_classes=class_count).to(
        dtype=support_embeddings.dtype
    )
    counts = membership.sum(dim=0)
    if bool((counts < 2).any().item()):
        raise OODCalibrationError("every traffic category requires at least two support examples")

    class_means = membership.transpose(0, 1) @ support_embeddings
    class_means = class_means / counts.unsqueeze(1)
    residuals = support_embeddings - class_means[support_labels]
    class_covariances = torch.einsum(
        "nk,nf,ng->kfg",
        membership,
        residuals,
        residuals,
    )
    class_covariances = class_covariances / counts[:, None, None]

    global_mean = support_embeddings.mean(dim=0)
    global_residuals = support_embeddings - global_mean
    global_covariance = global_residuals.transpose(0, 1) @ global_residuals
    global_covariance = global_covariance / support_embeddings.shape[0]

    tensors = (class_means, class_covariances, global_mean, global_covariance)
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise OODCalibrationError("relative Mahalanobis state must remain finite")
    return RelativeMahalanobisState(
        class_means=class_means.detach(),
        class_covariances=class_covariances.detach(),
        global_mean=global_mean.detach(),
        global_covariance=global_covariance.detach(),
    )


def relative_mahalanobis_scores(
    embeddings: Tensor,
    class_indices: Tensor,
    state: RelativeMahalanobisState,
) -> Tensor:
    """Return class distance minus global distance for each embedding."""
    _validate_queries(embeddings, class_indices, state)
    class_precision = torch.linalg.pinv(state.class_covariances, hermitian=True)
    global_precision = torch.linalg.pinv(state.global_covariance, hermitian=True)
    class_differences = embeddings - state.class_means[class_indices]
    global_differences = embeddings - state.global_mean
    selected_precision = class_precision[class_indices]
    class_distances = torch.einsum(
        "nf,nfg,ng->n",
        class_differences,
        selected_precision,
        class_differences,
    )
    global_distances = torch.einsum(
        "nf,fg,ng->n",
        global_differences,
        global_precision,
        global_differences,
    )
    scores = class_distances - global_distances
    if not bool(torch.isfinite(scores).all().item()):
        raise OODCalibrationError("relative Mahalanobis scores must remain finite")
    return scores


def fit_gaussian_kde(scores: Float64Vector) -> GaussianKDE1D:
    """Fit a univariate Gaussian KDE using the documented Scott bandwidth."""
    _validate_score_vector(scores, name="KDE samples")
    if scores.size < 2:
        raise OODCalibrationError("KDE fitting requires at least two samples")
    sample_deviation = float(np.std(scores, ddof=1))
    bandwidth = sample_deviation * float(scores.size ** (-1.0 / 5.0))
    if not isfinite(bandwidth) or bandwidth <= 0.0:
        raise OODCalibrationError("KDE samples must have positive finite variance")
    samples = np.asarray(scores, dtype=np.float64).copy()
    samples.setflags(write=False)
    return GaussianKDE1D(samples=samples, bandwidth=bandwidth)


def fit_class_conditional_kdes(
    relative_scores: Float64Vector,
    true_class_indices: npt.NDArray[np.int64],
) -> tuple[GaussianKDE1D, ...]:
    """Fit one relative-distance KDE per true calibration category."""
    _validate_score_vector(relative_scores, name="relative Mahalanobis scores")
    _validate_class_indices(true_class_indices, expected_length=relative_scores.size)
    return tuple(
        fit_gaussian_kde(relative_scores[true_class_indices == class_index])
        for class_index in range(len(CATEGORY_LABELS))
    )


def class_conditional_ood_scores(
    relative_scores: Float64Vector,
    predicted_class_indices: npt.NDArray[np.int64],
    class_kdes: tuple[GaussianKDE1D, ...],
) -> Float64Vector:
    """Return one minus each predicted class's fitted upper-tail p-value."""
    _validate_score_vector(relative_scores, name="relative Mahalanobis scores")
    _validate_class_indices(predicted_class_indices, expected_length=relative_scores.size)
    if len(class_kdes) != len(CATEGORY_LABELS):
        raise OODCalibrationError("one KDE is required for every traffic category")
    scores = np.empty(relative_scores.size, dtype=np.float64)
    for class_index, kde in enumerate(class_kdes):
        selected = predicted_class_indices == class_index
        if np.any(selected):
            scores[selected] = kde.cdf(relative_scores[selected])
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise OODCalibrationError("OOD scores must remain finite and within zero and one")
    scores.setflags(write=False)
    return scores


def _validate_support(embeddings: Tensor, labels: Tensor) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise OODCalibrationError("support embeddings must be nonempty and two-dimensional")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise OODCalibrationError(
            f"support embeddings must contain {EMBEDDING_DIMENSION} dimensions"
        )
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all().item()):
        raise OODCalibrationError("support embeddings must contain finite floating-point values")
    _validate_tensor_class_indices(labels, expected_length=embeddings.shape[0])


def _validate_queries(
    embeddings: Tensor,
    class_indices: Tensor,
    state: RelativeMahalanobisState,
) -> None:
    _validate_support_shape(embeddings)
    _validate_tensor_class_indices(class_indices, expected_length=embeddings.shape[0])
    class_count = len(CATEGORY_LABELS)
    expected_shapes = (
        (state.class_means, (class_count, EMBEDDING_DIMENSION)),
        (
            state.class_covariances,
            (class_count, EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),
        ),
        (state.global_mean, (EMBEDDING_DIMENSION,)),
        (state.global_covariance, (EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)),
    )
    if any(tuple(tensor.shape) != shape for tensor, shape in expected_shapes):
        raise OODCalibrationError("relative Mahalanobis state has invalid tensor shapes")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor, _ in expected_shapes):
        raise OODCalibrationError("relative Mahalanobis state must contain finite values")


def _validate_support_shape(embeddings: Tensor) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise OODCalibrationError("query embeddings must be nonempty and two-dimensional")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise OODCalibrationError(f"query embeddings must contain {EMBEDDING_DIMENSION} dimensions")
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all().item()):
        raise OODCalibrationError("query embeddings must contain finite floating-point values")


def _validate_tensor_class_indices(indices: Tensor, *, expected_length: int) -> None:
    if indices.ndim != 1 or indices.shape[0] != expected_length:
        raise OODCalibrationError("class indices must align one-to-one with embeddings")
    if indices.dtype != torch.int64:
        raise OODCalibrationError("class indices must use int64 values")
    if bool(((indices < 0) | (indices >= len(CATEGORY_LABELS))).any().item()):
        raise OODCalibrationError("class indices contain an unknown traffic category")


def _validate_score_vector(scores: Float64Vector, *, name: str) -> None:
    if scores.ndim != 1 or scores.size < 1:
        raise OODCalibrationError(f"{name} must be a nonempty one-dimensional array")
    if scores.dtype != np.float64 or not np.isfinite(scores).all():
        raise OODCalibrationError(f"{name} must contain finite float64 values")


def _validate_class_indices(
    indices: npt.NDArray[np.int64],
    *,
    expected_length: int,
) -> None:
    if indices.ndim != 1 or indices.size != expected_length:
        raise OODCalibrationError("class indices must align one-to-one with scores")
    if indices.dtype != np.int64:
        raise OODCalibrationError("class indices must use int64 values")
    if np.any((indices < 0) | (indices >= len(CATEGORY_LABELS))):
        raise OODCalibrationError("class indices contain an unknown traffic category")
