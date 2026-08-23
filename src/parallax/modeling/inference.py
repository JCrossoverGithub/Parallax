"""Post-training support statistics and validation-only prototype inference."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from parallax.data import DatasetPartition
from parallax.modeling.baselines import (
    CATEGORY_LABELS,
    BaselineModelingError,
    ClassificationMetrics,
    _classification_metrics,
    _validate_partition,
)
from parallax.modeling.dataset import FeaturePartition
from parallax.modeling.episodes import (
    CATEGORY_TO_INDEX,
    EpisodeSamplingError,
    _validate_training_partition,
)
from parallax.modeling.prototypes import calculate_class_prototypes
from parallax.modeling.training import PrototypeTrainingResult

PROTOTYPE_VALIDATION_SCHEMA_VERSION: Final = "vnat-prototype-validation-1"


class PrototypeInferenceError(ValueError):
    """Raised when post-training inference violates a partition or tensor contract."""


@dataclass(frozen=True, slots=True)
class PrototypeInferenceConfig:
    """Fixed support selection and validation-calibration diagnostic settings."""

    maximum_support_per_class: int = 100
    support_random_seed: int = 17
    expected_calibration_error_bins: int = 15

    def __post_init__(self) -> None:
        if self.maximum_support_per_class < 2:
            raise PrototypeInferenceError("maximum support per class must be at least two")
        if self.support_random_seed < 0:
            raise PrototypeInferenceError("support random seed must not be negative")
        if self.expected_calibration_error_bins < 2:
            raise PrototypeInferenceError("ECE bin count must be at least two")

    def as_dict(self) -> dict[str, object]:
        """Return the recorded closed-set inference policy."""
        return {
            "maximum_support_per_class": self.maximum_support_per_class,
            "support_random_seed": self.support_random_seed,
            "classification_distance": "diagonal-mahalanobis",
            "covariance_normalization": "population",
            "singular_covariance_handling": "torch-linalg-pseudoinverse",
            "expected_calibration_error_bins": self.expected_calibration_error_bins,
        }


@dataclass(frozen=True, slots=True)
class PrototypeInferenceState:
    """Training-derived class statistics used for closed-set inference."""

    support_indices: Tensor
    support_examples_per_class: tuple[int, ...]
    class_means: Tensor
    class_variances: Tensor


@dataclass(frozen=True, slots=True)
class PrototypeValidationResult:
    """Validation-only metrics for one trained prototype candidate."""

    config: PrototypeInferenceConfig
    inference_state: PrototypeInferenceState
    training: PrototypeTrainingResult
    validation_metrics: ClassificationMetrics
    expected_calibration_error: float
    validation_windows: int

    def as_dict(self) -> dict[str, object]:
        """Return stable closed-set validation evidence without calibration/test access."""
        return {
            "schema_version": PROTOTYPE_VALIDATION_SCHEMA_VERSION,
            "configuration": self.config.as_dict(),
            "training": self.training.as_dict(),
            "data": {
                "support_fit_partition": DatasetPartition.TRAIN.value,
                "evaluation_partition": DatasetPartition.VALIDATION.value,
                "validation_windows": self.validation_windows,
                "calibration_evaluated": False,
                "test_evaluated": False,
            },
            "category_order": list(CATEGORY_LABELS),
            "support_examples_per_class": {
                category: self.inference_state.support_examples_per_class[index]
                for index, category in enumerate(CATEGORY_LABELS)
            },
            "validation_metrics": self.validation_metrics.as_dict(),
            "expected_calibration_error": self.expected_calibration_error,
        }


def build_prototype_inference_state(
    training_result: PrototypeTrainingResult,
    training: FeaturePartition,
    *,
    config: PrototypeInferenceConfig | None = None,
) -> PrototypeInferenceState:
    """Sample training support and fit class means and diagonal covariances."""
    selected_config = config or PrototypeInferenceConfig()
    try:
        _validate_training_partition(training, training_result.config.episode)
    except EpisodeSamplingError as error:
        raise PrototypeInferenceError(f"invalid training partition: {error}") from error
    if training.windows != training_result.training_windows:
        raise PrototypeInferenceError("training result does not match training window count")

    labels = np.asarray(
        [CATEGORY_TO_INDEX[category] for category in training.categories],
        dtype=np.int64,
    )
    random = np.random.default_rng(selected_config.support_random_seed)
    selected_by_class = tuple(
        np.asarray(
            random.choice(
                np.flatnonzero(labels == class_index),
                size=min(
                    selected_config.maximum_support_per_class,
                    int(np.count_nonzero(labels == class_index)),
                ),
                replace=False,
            ),
            dtype=np.int64,
        )
        for class_index in range(len(CATEGORY_LABELS))
    )
    support_indices = np.concatenate(selected_by_class)
    support_labels = torch.from_numpy(labels[support_indices])
    standardized = training_result.standardize(training.features)
    support_features = torch.from_numpy(standardized[support_indices].copy())

    with torch.no_grad():
        support_embeddings = training_result.network(support_features)
        class_means = calculate_class_prototypes(support_embeddings, support_labels)
        residuals = support_embeddings - class_means[support_labels]
        membership = functional.one_hot(
            support_labels,
            num_classes=len(CATEGORY_LABELS),
        ).to(dtype=support_embeddings.dtype)
        counts = membership.sum(dim=0)
        class_variances = membership.transpose(0, 1) @ residuals.square()
        class_variances = class_variances / counts.unsqueeze(1)

    return PrototypeInferenceState(
        support_indices=torch.from_numpy(support_indices),
        support_examples_per_class=tuple(int(indices.size) for indices in selected_by_class),
        class_means=class_means.detach(),
        class_variances=class_variances.detach(),
    )


def prototype_class_probabilities(
    training_result: PrototypeTrainingResult,
    inference_state: PrototypeInferenceState,
    features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> Tensor:
    """Return paper-aligned diagonal-Mahalanobis class probabilities."""
    standardized = training_result.standardize(features)
    feature_tensor = torch.from_numpy(standardized.copy())
    with torch.no_grad():
        embeddings = training_result.network(feature_tensor)
        differences = embeddings[:, None, :] - inference_state.class_means[None, :, :]
        covariance = torch.diag_embed(inference_state.class_variances)
        precision = torch.linalg.pinv(covariance, hermitian=True)
        distances = torch.einsum(
            "nkf,kfg,nkg->nk",
            differences,
            precision,
            differences,
        )
        probabilities = functional.softmax(-distances, dim=1)
    if probabilities.shape[1] != len(CATEGORY_LABELS):
        raise PrototypeInferenceError("inference state must contain every traffic category")
    if not bool(torch.isfinite(probabilities).all().item()):
        raise PrototypeInferenceError("prototype probabilities must remain finite")
    return probabilities


def evaluate_prototype_validation(
    training_result: PrototypeTrainingResult,
    training: FeaturePartition,
    validation: FeaturePartition,
    *,
    config: PrototypeInferenceConfig | None = None,
) -> PrototypeValidationResult:
    """Fit support on training and evaluate the candidate on validation only."""
    selected_config = config or PrototypeInferenceConfig()
    try:
        _validate_partition(validation, expected=DatasetPartition.VALIDATION)
    except BaselineModelingError as error:
        raise PrototypeInferenceError(f"invalid validation partition: {error}") from error
    if set(training.capture_ids).intersection(validation.capture_ids):
        raise PrototypeInferenceError("training and validation captures must not overlap")

    state = build_prototype_inference_state(
        training_result,
        training,
        config=selected_config,
    )
    probabilities = prototype_class_probabilities(
        training_result,
        state,
        validation.features,
    )
    predicted_indices = probabilities.argmax(dim=1).cpu().numpy()
    predictions = np.asarray(
        [CATEGORY_LABELS[int(index)] for index in predicted_indices],
        dtype=np.str_,
    )
    expected_indices = np.asarray(
        [CATEGORY_TO_INDEX[category] for category in validation.categories],
        dtype=np.int64,
    )
    return PrototypeValidationResult(
        config=selected_config,
        inference_state=state,
        training=training_result,
        validation_metrics=_classification_metrics(validation.categories, predictions),
        expected_calibration_error=_expected_calibration_error(
            probabilities.cpu().numpy(),
            expected_indices,
            bins=selected_config.expected_calibration_error_bins,
        ),
        validation_windows=validation.windows,
    )


def _expected_calibration_error(
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    expected_indices: np.ndarray[tuple[int], np.dtype[np.int64]],
    *,
    bins: int,
) -> float:
    confidences = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == expected_indices
    bin_indices = np.minimum((confidences * bins).astype(np.int64), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        if np.any(selected):
            weight = float(np.count_nonzero(selected) / probabilities.shape[0])
            accuracy = float(np.mean(correct[selected]))
            confidence = float(np.mean(confidences[selected]))
            error += weight * abs(accuracy - confidence)
    return error
