from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from parallax.data import FEATURE_COUNT
from parallax.features.runtime import VnatWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.bundle import LoadedPrototypeBundle
from parallax.modeling.calibration_bundle import LoadedPrototypeCalibration
from parallax.modeling.runtime import (
    PrototypeRuntime,
    load_prototype_runtime,
)
from parallax.modeling.scoring import PrototypeScores

MODEL_SHA256 = "a" * 64
CALIBRATION_SHA256 = "b" * 64
FEATURE_SHA256 = "c" * 64
SPLIT_SHA256 = "d" * 64


def _bundle() -> LoadedPrototypeBundle:
    return cast(
        LoadedPrototypeBundle,
        SimpleNamespace(
            sha256=MODEL_SHA256,
            feature_artifact_sha256=FEATURE_SHA256,
            split_manifest_sha256=SPLIT_SHA256,
        ),
    )


def _calibration() -> LoadedPrototypeCalibration:
    return cast(
        LoadedPrototypeCalibration,
        SimpleNamespace(
            sha256=CALIBRATION_SHA256,
            model_bundle_sha256=MODEL_SHA256,
            feature_artifact_sha256=FEATURE_SHA256,
            split_manifest_sha256=SPLIT_SHA256,
        ),
    )


def _feature() -> VnatWindowFeature:
    values = np.arange(FEATURE_COUNT, dtype=np.float32)
    values.setflags(write=False)

    return cast(
        VnatWindowFeature,
        SimpleNamespace(
            window_id="capture:flow:0",
            capture=SimpleNamespace(capture_id="nonvpn_ssh_capture4.pcap"),
            flow_id="flow",
            window_index=0,
            start_offset_seconds=0.0,
            end_offset_seconds=40.96,
            packet_count=21,
            values=values,
        ),
    )


def test_scores_one_runtime_feature_with_traceable_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(
        bundle: LoadedPrototypeBundle,
        calibration: LoadedPrototypeCalibration,
        features: np.ndarray,
    ) -> PrototypeScores:
        assert bundle is runtime.bundle
        assert calibration is runtime.calibration
        assert features.shape == (1, FEATURE_COUNT)
        assert features.dtype == np.float32

        probabilities = np.asarray(
            [[0.05, 0.10, 0.15, 0.60, 0.10]],
            dtype=np.float32,
        )
        predicted_indices = np.asarray([3], dtype=np.int64)
        predicted_categories = np.asarray(
            [CATEGORY_LABELS[3]],
            dtype=np.str_,
        )
        relative_distances = np.asarray([1.25], dtype=np.float64)
        ood_scores = np.asarray([0.2], dtype=np.float64)

        return PrototypeScores(
            category_order=CATEGORY_LABELS,
            class_probabilities=probabilities,
            predicted_class_indices=predicted_indices,
            predicted_categories=predicted_categories,
            relative_mahalanobis_distances=relative_distances,
            ood_scores=ood_scores,
        )

    runtime = PrototypeRuntime(
        bundle=_bundle(),
        calibration=_calibration(),
    )

    monkeypatch.setattr(
        "parallax.modeling.runtime.score_prototype_ood",
        fake_score,
    )

    prediction = runtime.score_feature(_feature())

    assert prediction.window_id == "capture:flow:0"
    assert prediction.capture_id == "nonvpn_ssh_capture4.pcap"
    assert prediction.flow_id == "flow"
    assert prediction.window_index == 0
    assert prediction.start_offset_seconds == 0.0
    assert prediction.end_offset_seconds == 40.96
    assert prediction.packet_count == 21
    assert prediction.category_order == CATEGORY_LABELS
    assert prediction.predicted_class_index == 3
    assert prediction.predicted_category == CATEGORY_LABELS[3]
    assert prediction.raw_confidence == pytest.approx(0.6)
    assert prediction.relative_mahalanobis_distance == 1.25
    assert prediction.ood_score == 0.2
    assert prediction.model_bundle_sha256 == MODEL_SHA256
    assert prediction.calibration_artifact_sha256 == CALIBRATION_SHA256
    assert prediction.feature_artifact_sha256 == FEATURE_SHA256
    assert prediction.split_manifest_sha256 == SPLIT_SHA256
    assert prediction.class_probabilities == pytest.approx((0.05, 0.10, 0.15, 0.60, 0.10))


def test_loads_mutually_bound_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    calibration = _calibration()
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_model_loader(
        path: str | Path,
        **kwargs: Any,
    ) -> LoadedPrototypeBundle:
        calls.append((str(path), kwargs))
        return bundle

    def fake_calibration_loader(
        path: str | Path,
        **kwargs: Any,
    ) -> LoadedPrototypeCalibration:
        calls.append((str(path), kwargs))
        return calibration

    monkeypatch.setattr(
        "parallax.modeling.runtime.load_prototype_model_bundle",
        fake_model_loader,
    )
    monkeypatch.setattr(
        "parallax.modeling.runtime.load_prototype_ood_calibration",
        fake_calibration_loader,
    )

    model_path = tmp_path / "model.json"
    calibration_path = tmp_path / "calibration.json"

    runtime = load_prototype_runtime(
        model_path,
        calibration_path,
        expected_model_bundle_sha256=MODEL_SHA256,
        expected_calibration_artifact_sha256=CALIBRATION_SHA256,
        expected_feature_artifact_sha256=FEATURE_SHA256,
        expected_split_manifest_sha256=SPLIT_SHA256,
    )

    assert runtime.bundle is bundle
    assert runtime.calibration is calibration
    assert calls == [
        (
            str(model_path),
            {
                "expected_sha256": MODEL_SHA256,
                "expected_feature_artifact_sha256": FEATURE_SHA256,
                "expected_split_manifest_sha256": SPLIT_SHA256,
            },
        ),
        (
            str(calibration_path),
            {
                "expected_sha256": CALIBRATION_SHA256,
                "expected_model_bundle_sha256": MODEL_SHA256,
                "expected_feature_artifact_sha256": FEATURE_SHA256,
                "expected_split_manifest_sha256": SPLIT_SHA256,
            },
        ),
    ]
