"""Stable operator-facing events emitted by the Parallax runtime."""

from dataclasses import dataclass
from typing import Final

from parallax.modeling.runtime import PrototypeRuntimePrediction

RUNTIME_PREDICTION_EVENT_SCHEMA_VERSION: Final = "parallax-runtime-prediction-1"


@dataclass(frozen=True, slots=True)
class RuntimePredictionEvent:
    """Serializable prediction event for downstream operator interfaces."""

    run_id: str
    prediction: PrototypeRuntimePrediction

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("runtime prediction event run ID must not be empty")

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible runtime event."""
        prediction = self.prediction

        return {
            "schema_version": RUNTIME_PREDICTION_EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "window": {
                "window_id": prediction.window_id,
                "capture_id": prediction.capture_id,
                "flow_id": prediction.flow_id,
                "window_index": prediction.window_index,
                "start_offset_seconds": prediction.start_offset_seconds,
                "end_offset_seconds": prediction.end_offset_seconds,
                "packet_count": prediction.packet_count,
            },
            "classification": {
                "category_order": list(prediction.category_order),
                "class_probabilities": list(prediction.class_probabilities),
                "predicted_class_index": prediction.predicted_class_index,
                "predicted_category": prediction.predicted_category,
                "raw_confidence": prediction.raw_confidence,
            },
            "uncertainty": {
                "relative_mahalanobis_distance": (prediction.relative_mahalanobis_distance),
                "ood_score": prediction.ood_score,
            },
            "provenance": {
                "model_bundle_sha256": prediction.model_bundle_sha256,
                "calibration_artifact_sha256": (prediction.calibration_artifact_sha256),
                "feature_artifact_sha256": prediction.feature_artifact_sha256,
                "split_manifest_sha256": prediction.split_manifest_sha256,
            },
        }
