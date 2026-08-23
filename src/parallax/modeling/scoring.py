"""Deterministic closed-set and OOD scoring from frozen JSON state."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional

from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.bundle import LoadedPrototypeBundle, PrototypeBundleError
from parallax.modeling.calibration_bundle import LoadedPrototypeCalibration
from parallax.modeling.uncertainty import (
    OODCalibrationError,
    class_conditional_ood_scores,
    relative_mahalanobis_scores,
)

Float32Matrix = npt.NDArray[np.float32]
Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
StringVector = npt.NDArray[np.str_]


class PrototypeScoringError(ValueError):
    """Raised when frozen prototype scoring violates an artifact contract."""


@dataclass(frozen=True, slots=True)
class PrototypeScores:
    """Immutable classification and OOD outputs for an unlabeled feature matrix."""

    category_order: tuple[str, ...]
    class_probabilities: Float32Matrix
    predicted_class_indices: Int64Vector
    predicted_categories: StringVector
    relative_mahalanobis_distances: Float64Vector
    ood_scores: Float64Vector


def score_prototype_ood(
    bundle: LoadedPrototypeBundle,
    calibration: LoadedPrototypeCalibration,
    features: Float32Matrix,
) -> PrototypeScores:
    """Score unlabeled features using mutually bound model and calibration state."""
    _validate_binding(bundle, calibration)
    try:
        standardized = bundle.standardize(features)
        feature_tensor = torch.from_numpy(standardized.copy())
        with torch.no_grad():
            embeddings = bundle.network(feature_tensor)
            differences = embeddings[:, None, :] - bundle.inference_state.class_means[None, :, :]
            covariance = torch.diag_embed(bundle.inference_state.class_variances)
            precision = torch.linalg.pinv(covariance, hermitian=True)
            distances = torch.einsum(
                "nkf,kfg,nkg->nk",
                differences,
                precision,
                differences,
            )
            probabilities = functional.softmax(-distances, dim=1)
            predicted_tensor = probabilities.argmax(dim=1)
            relative_tensor = relative_mahalanobis_scores(
                embeddings,
                predicted_tensor,
                calibration.geometry,
            )
        predicted_indices = np.asarray(predicted_tensor.cpu().numpy(), dtype=np.int64)
        relative_distances = np.asarray(relative_tensor.cpu().numpy(), dtype=np.float64)
        ood_scores = class_conditional_ood_scores(
            relative_distances,
            predicted_indices,
            calibration.class_kdes,
        )
    except (OODCalibrationError, PrototypeBundleError, RuntimeError) as error:
        raise PrototypeScoringError(f"prototype scoring failed: {error}") from error

    class_probabilities = np.asarray(probabilities.cpu().numpy(), dtype=np.float32)
    if class_probabilities.shape != (features.shape[0], len(CATEGORY_LABELS)):
        raise PrototypeScoringError("prototype probabilities have an invalid shape")
    if not np.isfinite(class_probabilities).all():
        raise PrototypeScoringError("prototype probabilities must remain finite")
    predicted_categories = np.asarray(
        [CATEGORY_LABELS[int(index)] for index in predicted_indices],
        dtype=np.str_,
    )
    for array in (
        class_probabilities,
        predicted_indices,
        predicted_categories,
        relative_distances,
    ):
        array.setflags(write=False)
    return PrototypeScores(
        category_order=CATEGORY_LABELS,
        class_probabilities=class_probabilities,
        predicted_class_indices=predicted_indices,
        predicted_categories=predicted_categories,
        relative_mahalanobis_distances=relative_distances,
        ood_scores=ood_scores,
    )


def _validate_binding(
    bundle: LoadedPrototypeBundle,
    calibration: LoadedPrototypeCalibration,
) -> None:
    if calibration.model_bundle_sha256 != bundle.sha256:
        raise PrototypeScoringError("calibration artifact does not match the model bundle")
    if calibration.feature_artifact_sha256 != bundle.feature_artifact_sha256:
        raise PrototypeScoringError("calibration and model feature provenance do not match")
    if calibration.split_manifest_sha256 != bundle.split_manifest_sha256:
        raise PrototypeScoringError("calibration and model split provenance do not match")
