import pytest

from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.runtime import (
    RUNTIME_PREDICTION_EVENT_SCHEMA_VERSION,
    RuntimePredictionEvent,
)


def prediction() -> PrototypeRuntimePrediction:
    return PrototypeRuntimePrediction(
        window_id="capture:flow:3",
        capture_id="nonvpn_ssh_capture4.pcap",
        flow_id="flow",
        window_index=3,
        start_offset_seconds=122.88,
        end_offset_seconds=163.84,
        packet_count=42,
        category_order=CATEGORY_LABELS,
        class_probabilities=(0.1, 0.2, 0.3, 0.25, 0.15),
        predicted_class_index=2,
        predicted_category=CATEGORY_LABELS[2],
        raw_confidence=0.3,
        relative_mahalanobis_distance=1.75,
        ood_score=0.125,
        model_bundle_sha256="a" * 64,
        calibration_artifact_sha256="b" * 64,
        feature_artifact_sha256="c" * 64,
        split_manifest_sha256="d" * 64,
    )


def test_serializes_runtime_prediction_event() -> None:
    event = RuntimePredictionEvent(
        run_id="replay-001",
        prediction=prediction(),
    )

    assert event.as_dict() == {
        "schema_version": RUNTIME_PREDICTION_EVENT_SCHEMA_VERSION,
        "run_id": "replay-001",
        "window": {
            "window_id": "capture:flow:3",
            "capture_id": "nonvpn_ssh_capture4.pcap",
            "flow_id": "flow",
            "window_index": 3,
            "start_offset_seconds": 122.88,
            "end_offset_seconds": 163.84,
            "packet_count": 42,
        },
        "classification": {
            "category_order": list(CATEGORY_LABELS),
            "class_probabilities": [0.1, 0.2, 0.3, 0.25, 0.15],
            "predicted_class_index": 2,
            "predicted_category": CATEGORY_LABELS[2],
            "raw_confidence": 0.3,
        },
        "uncertainty": {
            "relative_mahalanobis_distance": 1.75,
            "ood_score": 0.125,
        },
        "provenance": {
            "model_bundle_sha256": "a" * 64,
            "calibration_artifact_sha256": "b" * 64,
            "feature_artifact_sha256": "c" * 64,
            "split_manifest_sha256": "d" * 64,
        },
    }


def test_rejects_empty_runtime_run_id() -> None:
    with pytest.raises(ValueError, match="run ID must not be empty"):
        RuntimePredictionEvent(
            run_id="",
            prediction=prediction(),
        )
