"""Runtime scoring from trusted frozen prototype artifacts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parallax.features.runtime import VnatWindowFeature
from parallax.modeling.bundle import (
    LoadedPrototypeBundle,
    load_prototype_model_bundle,
)
from parallax.modeling.calibration_bundle import (
    LoadedPrototypeCalibration,
    load_prototype_ood_calibration,
)
from parallax.modeling.scoring import score_prototype_ood


@dataclass(frozen=True, slots=True)
class PrototypeRuntimePrediction:
    """One traceable runtime prediction from frozen prototype artifacts."""

    window_id: str
    capture_id: str
    flow_id: str
    window_index: int
    start_offset_seconds: float
    end_offset_seconds: float
    packet_count: int
    category_order: tuple[str, ...]
    class_probabilities: tuple[float, ...]
    predicted_class_index: int
    predicted_category: str
    raw_confidence: float
    relative_mahalanobis_distance: float
    ood_score: float
    model_bundle_sha256: str
    calibration_artifact_sha256: str
    feature_artifact_sha256: str
    split_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class PrototypeRuntime:
    """Mutually bound frozen model and OOD calibration state."""

    bundle: LoadedPrototypeBundle
    calibration: LoadedPrototypeCalibration

    def score_feature(
        self,
        feature: VnatWindowFeature,
    ) -> PrototypeRuntimePrediction:
        """Score one runtime feature without using capture labels."""
        feature_matrix = np.asarray(feature.values, dtype=np.float32)[None, :]
        scores = score_prototype_ood(
            self.bundle,
            self.calibration,
            feature_matrix,
        )

        predicted_index = int(scores.predicted_class_indices[0])
        probabilities = tuple(float(value) for value in scores.class_probabilities[0])

        return PrototypeRuntimePrediction(
            window_id=feature.window_id,
            capture_id=feature.capture.capture_id,
            flow_id=feature.flow_id,
            window_index=feature.window_index,
            start_offset_seconds=feature.start_offset_seconds,
            end_offset_seconds=feature.end_offset_seconds,
            packet_count=feature.packet_count,
            category_order=scores.category_order,
            class_probabilities=probabilities,
            predicted_class_index=predicted_index,
            predicted_category=str(scores.predicted_categories[0]),
            raw_confidence=probabilities[predicted_index],
            relative_mahalanobis_distance=float(scores.relative_mahalanobis_distances[0]),
            ood_score=float(scores.ood_scores[0]),
            model_bundle_sha256=self.bundle.sha256,
            calibration_artifact_sha256=self.calibration.sha256,
            feature_artifact_sha256=self.bundle.feature_artifact_sha256,
            split_manifest_sha256=self.bundle.split_manifest_sha256,
        )


def load_prototype_runtime(
    model_bundle: str | Path,
    calibration_artifact: str | Path,
    *,
    expected_model_bundle_sha256: str,
    expected_calibration_artifact_sha256: str,
    expected_feature_artifact_sha256: str,
    expected_split_manifest_sha256: str,
) -> PrototypeRuntime:
    """Load mutually bound frozen prototype artifacts for runtime inference."""
    bundle = load_prototype_model_bundle(
        model_bundle,
        expected_sha256=expected_model_bundle_sha256,
        expected_feature_artifact_sha256=expected_feature_artifact_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    calibration = load_prototype_ood_calibration(
        calibration_artifact,
        expected_sha256=expected_calibration_artifact_sha256,
        expected_model_bundle_sha256=bundle.sha256,
        expected_feature_artifact_sha256=expected_feature_artifact_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    return PrototypeRuntime(
        bundle=bundle,
        calibration=calibration,
    )
