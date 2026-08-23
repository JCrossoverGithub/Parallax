"""Tests for immutable one-shot prototype test reports."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from parallax.data import DatasetPartition
from parallax.modeling import (
    CATEGORY_LABELS,
    MODELING_DATASET_SCHEMA_VERSION,
    PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION,
    PROTOTYPE_TEST_REPORT_SCHEMA_VERSION,
    CategoryMetrics,
    ClassificationMetrics,
    OODCategoryRate,
    OODScoreSummary,
    OODThresholdResult,
    PartitionedFeatureDataset,
    PrototypeTestEvaluation,
    PrototypeTestReportError,
    export_prototype_test_report,
)


def evaluation_fixture() -> PrototypeTestEvaluation:
    category_metrics = {
        category: CategoryMetrics(precision=1.0, recall=1.0, f1=1.0, support=2)
        for category in CATEGORY_LABELS
    }
    threshold_categories = {
        category: OODCategoryRate(windows=2, flagged_windows=0, false_positive_rate=0.0)
        for category in CATEGORY_LABELS
    }
    return PrototypeTestEvaluation(
        classification_metrics=ClassificationMetrics(
            accuracy=1.0,
            balanced_accuracy=1.0,
            micro_f1=1.0,
            macro_f1=1.0,
            categories=category_metrics,
            confusion_matrix=tuple(
                tuple(2 if row == column else 0 for column in range(5)) for row in range(5)
            ),
        ),
        expected_calibration_error=0.01,
        ood_score_summary=OODScoreSummary(
            minimum=0.01,
            quantile_05=0.02,
            median=0.5,
            quantile_95=0.9,
            quantile_99=0.94,
            maximum=0.949,
            mean=0.5,
        ),
        ood_thresholds=(
            OODThresholdResult(
                threshold=0.95,
                flagged_windows=0,
                false_positive_rate=0.0,
                categories=threshold_categories,
            ),
            OODThresholdResult(
                threshold=0.99,
                flagged_windows=0,
                false_positive_rate=0.0,
                categories=threshold_categories,
            ),
        ),
        test_windows=10,
        test_captures=5,
    )


def test_exports_replayable_provenance_bound_test_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    bundle_path = tmp_path / "prototype-model.json"
    calibration_path = tmp_path / "prototype-calibration.json"
    for path, content in (
        (features, b"features"),
        (manifest, b"manifest"),
        (bundle_path, b"bundle"),
        (calibration_path, b"calibration"),
    ):
        path.write_bytes(content)

    dataset = cast(
        "PartitionedFeatureDataset",
        SimpleNamespace(
            feature_artifact_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
        ),
    )
    loaded_bundle = SimpleNamespace(sha256="c" * 64)
    loaded_calibration = SimpleNamespace(sha256="d" * 64)
    evaluation = evaluation_fixture()
    calls: list[object] = []

    def fake_dataset(
        selected_features: Path,
        selected_manifest: Path,
        *,
        expected_manifest_sha256: str,
    ) -> PartitionedFeatureDataset:
        calls.append((selected_features, selected_manifest, expected_manifest_sha256))
        return dataset

    def fake_bundle(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return loaded_bundle

    def fake_calibration(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return loaded_calibration

    def fake_partition(partition: DatasetPartition) -> object:
        assert partition is DatasetPartition.TEST
        return "test-partition"

    def fake_evaluate(*args: object) -> PrototypeTestEvaluation:
        calls.append(args)
        return evaluation

    monkeypatch.setattr(
        "parallax.modeling.final_report.load_partitioned_feature_dataset",
        fake_dataset,
    )
    monkeypatch.setattr(
        "parallax.modeling.final_report.load_prototype_model_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "parallax.modeling.final_report.load_prototype_ood_calibration",
        fake_calibration,
    )
    monkeypatch.setattr(dataset, "partition", fake_partition, raising=False)
    monkeypatch.setattr("parallax.modeling.final_report.evaluate_prototype_test", fake_evaluate)

    first_path = tmp_path / "first" / "prototype-test.json"
    replay_path = tmp_path / "replay" / "prototype-test.json"
    first = export_prototype_test_report(
        features,
        manifest,
        bundle_path,
        calibration_path,
        first_path,
        expected_manifest_sha256="b" * 64,
        expected_model_bundle_sha256="c" * 64,
        expected_calibration_artifact_sha256="d" * 64,
    )
    replay = export_prototype_test_report(
        features,
        manifest,
        bundle_path,
        calibration_path,
        replay_path,
        expected_manifest_sha256="b" * 64,
        expected_model_bundle_sha256="c" * 64,
        expected_calibration_artifact_sha256="d" * 64,
    )

    assert first_path.read_bytes() == replay_path.read_bytes()
    assert first.as_dict() == replay.as_dict()
    assert first.output_sha256 == sha256(first_path.read_bytes()).hexdigest()
    assert first.output_file_size_bytes == first_path.stat().st_size
    assert first.feature_artifact == str(features)
    assert first.split_manifest == str(manifest)
    assert first.model_bundle == str(bundle_path)
    assert first.calibration_artifact == str(calibration_path)
    assert len(calls) == 8

    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PROTOTYPE_TEST_REPORT_SCHEMA_VERSION
    assert payload["test_evaluation_schema_version"] == (PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION)
    assert payload["provenance"]["modeling_dataset_schema_version"] == (
        MODELING_DATASET_SCHEMA_VERSION
    )
    assert payload["provenance"]["model_bundle"] == {
        "file_name": bundle_path.name,
        "file_size_bytes": len(b"bundle"),
        "sha256": "c" * 64,
        "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    }
    assert payload["provenance"]["calibration_artifact"] == {
        "file_name": calibration_path.name,
        "file_size_bytes": len(b"calibration"),
        "sha256": "d" * 64,
        "schema_version": PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    }
    assert payload["evaluation_policy"] == {
        "candidate_selection_partition": "validation",
        "geometry_fit_partition": "train",
        "density_fit_partition": "calibration",
        "evaluation_partition": "test",
        "candidate_frozen_before_test": True,
        "model_selection_after_test": False,
        "test_evaluated": True,
    }


@pytest.mark.parametrize(
    ("name", "existing", "message"),
    [
        ("prototype-test.txt", False, "must use the .json extension"),
        ("prototype-test.json", True, "already exists"),
    ],
)
def test_rejects_invalid_destination(
    tmp_path: Path,
    name: str,
    existing: bool,
    message: str,
) -> None:
    output = tmp_path / name
    if existing:
        output.write_text("existing", encoding="utf-8")

    with pytest.raises(PrototypeTestReportError, match=message):
        export_prototype_test_report(
            tmp_path / "features.parquet",
            tmp_path / "manifest.json",
            tmp_path / "model.json",
            tmp_path / "calibration.json",
            output,
            expected_manifest_sha256="b" * 64,
            expected_model_bundle_sha256="c" * 64,
            expected_calibration_artifact_sha256="d" * 64,
        )


def test_wraps_test_report_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr(
        "parallax.modeling.final_report.load_partitioned_feature_dataset",
        fail,
    )

    with pytest.raises(PrototypeTestReportError, match="report failed: unavailable"):
        export_prototype_test_report(
            tmp_path / "features.parquet",
            tmp_path / "manifest.json",
            tmp_path / "model.json",
            tmp_path / "calibration.json",
            tmp_path / "test.json",
            expected_manifest_sha256="b" * 64,
            expected_model_bundle_sha256="c" * 64,
            expected_calibration_artifact_sha256="d" * 64,
        )
