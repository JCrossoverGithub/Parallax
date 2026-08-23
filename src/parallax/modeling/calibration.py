"""Calibration-only fitting of relative-Mahalanobis OOD distributions."""

from dataclasses import dataclass

import numpy as np
import torch

from parallax.data import DatasetPartition
from parallax.modeling.baselines import (
    CATEGORY_LABELS,
    BaselineModelingError,
    _validate_partition,
)
from parallax.modeling.bundle import LoadedPrototypeBundle, PrototypeBundleError
from parallax.modeling.dataset import FeaturePartition
from parallax.modeling.episodes import CATEGORY_TO_INDEX
from parallax.modeling.uncertainty import (
    KDE_BANDWIDTH_METHOD,
    OOD_CALIBRATION_SCHEMA_VERSION,
    GaussianKDE1D,
    OODCalibrationError,
    RelativeMahalanobisState,
    fit_class_conditional_kdes,
    fit_relative_mahalanobis_state,
    relative_mahalanobis_scores,
)


class PrototypeCalibrationError(ValueError):
    """Raised when calibration would violate the frozen-model partition contract."""


@dataclass(frozen=True, slots=True)
class PrototypeCalibrationResult:
    """Frozen-model geometry and calibration-only class density estimates."""

    bundle: LoadedPrototypeBundle
    geometry: RelativeMahalanobisState
    class_kdes: tuple[GaussianKDE1D, ...]
    relative_scores: np.ndarray[tuple[int], np.dtype[np.float64]]
    calibration_examples_per_class: tuple[int, ...]
    calibration_windows: int

    def as_dict(self) -> dict[str, object]:
        """Return stable evidence that calibration, not evaluation, was performed."""
        return {
            "schema_version": OOD_CALIBRATION_SCHEMA_VERSION,
            "model_bundle_sha256": self.bundle.sha256,
            "configuration": {
                "support_covariance": "full-population",
                "singular_covariance_handling": "torch-linalg-pseudoinverse",
                "relative_distance": "class-mahalanobis-minus-global-mahalanobis",
                "density_estimator": "class-conditional-univariate-gaussian-kde",
                "bandwidth_method": KDE_BANDWIDTH_METHOD,
                "p_value": "fitted-upper-tail-probability",
                "ood_score": "one-minus-p-value",
            },
            "data": {
                "geometry_fit_partition": DatasetPartition.TRAIN.value,
                "density_fit_partition": DatasetPartition.CALIBRATION.value,
                "calibration_windows": self.calibration_windows,
                "calibration_labels_used": True,
                "model_selection_performed": False,
                "test_evaluated": False,
            },
            "category_order": list(CATEGORY_LABELS),
            "calibration_examples_per_class": {
                category: self.calibration_examples_per_class[index]
                for index, category in enumerate(CATEGORY_LABELS)
            },
            "relative_score_summary": {
                category: {
                    "minimum": float(kde.samples.min()),
                    "maximum": float(kde.samples.max()),
                    "mean": float(kde.samples.mean()),
                    "kde": kde.as_dict(),
                }
                for category, kde in zip(CATEGORY_LABELS, self.class_kdes, strict=True)
            },
        }


def calibrate_prototype_ood(
    bundle: LoadedPrototypeBundle,
    training: FeaturePartition,
    calibration: FeaturePartition,
) -> PrototypeCalibrationResult:
    """Fit paper-aligned OOD KDEs using training support and calibration only."""
    try:
        _validate_partition(training, expected=DatasetPartition.TRAIN)
        _validate_partition(calibration, expected=DatasetPartition.CALIBRATION)
        if set(training.capture_ids).intersection(calibration.capture_ids):
            raise PrototypeCalibrationError("training and calibration captures must not overlap")

        support_embeddings, support_labels = _reconstruct_support(bundle, training)
        geometry = fit_relative_mahalanobis_state(support_embeddings, support_labels)
        _validate_reconstructed_statistics(bundle, geometry)

        calibration_labels = np.asarray(
            [CATEGORY_TO_INDEX[category] for category in calibration.categories],
            dtype=np.int64,
        )
        standardized = bundle.standardize(calibration.features)
        with torch.no_grad():
            calibration_embeddings = bundle.network(torch.from_numpy(standardized.copy()))
            score_tensor = relative_mahalanobis_scores(
                calibration_embeddings,
                torch.from_numpy(calibration_labels),
                geometry,
            )
        relative_scores = np.asarray(score_tensor.cpu().numpy(), dtype=np.float64)
        relative_scores.setflags(write=False)
        class_kdes = fit_class_conditional_kdes(relative_scores, calibration_labels)
    except (
        BaselineModelingError,
        OODCalibrationError,
        PrototypeBundleError,
        RuntimeError,
    ) as error:
        raise PrototypeCalibrationError(f"prototype OOD calibration failed: {error}") from error

    return PrototypeCalibrationResult(
        bundle=bundle,
        geometry=geometry,
        class_kdes=class_kdes,
        relative_scores=relative_scores,
        calibration_examples_per_class=tuple(
            int(np.count_nonzero(calibration_labels == class_index))
            for class_index in range(len(CATEGORY_LABELS))
        ),
        calibration_windows=calibration.windows,
    )


def _reconstruct_support(
    bundle: LoadedPrototypeBundle,
    training: FeaturePartition,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = bundle.inference_state.support_indices.cpu().numpy()
    if indices.size < 1 or int(indices.max()) >= training.windows:
        raise PrototypeCalibrationError("bundle support indices exceed the training partition")
    labels = np.asarray(
        [CATEGORY_TO_INDEX[category] for category in training.categories],
        dtype=np.int64,
    )
    support_labels = labels[indices]
    observed_counts = tuple(
        int(np.count_nonzero(support_labels == class_index))
        for class_index in range(len(CATEGORY_LABELS))
    )
    if observed_counts != bundle.inference_state.support_examples_per_class:
        raise PrototypeCalibrationError("bundle support indices do not match training categories")
    standardized = bundle.standardize(training.features[indices].copy())
    with torch.no_grad():
        embeddings = bundle.network(torch.from_numpy(standardized.copy()))
    return embeddings, torch.from_numpy(support_labels)


def _validate_reconstructed_statistics(
    bundle: LoadedPrototypeBundle,
    geometry: RelativeMahalanobisState,
) -> None:
    stored = bundle.inference_state
    reconstructed_variances = torch.diagonal(
        geometry.class_covariances,
        dim1=1,
        dim2=2,
    )
    if not torch.allclose(geometry.class_means, stored.class_means):
        raise PrototypeCalibrationError("bundle class means do not reproduce from training support")
    if not torch.allclose(reconstructed_variances, stored.class_variances):
        raise PrototypeCalibrationError(
            "bundle class variances do not reproduce from training support"
        )
