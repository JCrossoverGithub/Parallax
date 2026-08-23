"""Tests for immutable prototype model bundles and validation reports."""

import json
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

import parallax.modeling.prototype_report as report_module
from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    MODELING_DATASET_SCHEMA_VERSION,
    PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    PROTOTYPE_MODEL_SCHEMA_VERSION,
    PROTOTYPE_TRAINING_SCHEMA_VERSION,
    PROTOTYPE_VALIDATION_REPORT_SCHEMA_VERSION,
    PROTOTYPE_VALIDATION_SCHEMA_VERSION,
    EpisodeConfig,
    FeaturePartition,
    ModelingDatasetError,
    PartitionedFeatureDataset,
    PrototypeInferenceConfig,
    PrototypeTrainingConfig,
    PrototypeValidationReportError,
    export_prototype_validation_report,
)


def partition_fixture(partition: DatasetPartition, *, prefix: str) -> FeaturePartition:
    samples_per_category = 8
    row_count = len(TrafficCategory) * samples_per_category
    features = np.zeros((row_count, FEATURE_COUNT), dtype=np.float32)
    labels: list[str] = []
    captures: list[str] = []
    vpn_statuses: list[str] = []
    applications: list[str] = []
    row = 0
    for category_index, category in enumerate(TrafficCategory):
        for sample_index in range(samples_per_category):
            features[row, 0] = float(category_index * 20 + sample_index)
            features[row, category_index + 1] = float(sample_index + 1)
            labels.append(category.value)
            captures.append(f"{prefix}-{category.value}-{sample_index}.pcap")
            vpn_statuses.append("vpn" if sample_index % 2 else "nonvpn")
            applications.append(f"application-{category_index}")
            row += 1
    arrays = (
        features,
        np.asarray(labels, dtype=np.str_),
        np.asarray(captures, dtype=np.str_),
        np.asarray(vpn_statuses, dtype=np.str_),
        np.asarray(applications, dtype=np.str_),
    )
    for array in arrays:
        array.setflags(write=False)
    return FeaturePartition(
        partition=partition,
        features=arrays[0],
        categories=arrays[1],
        capture_ids=arrays[2],
        vpn_statuses=arrays[3],
        applications=arrays[4],
    )


def dataset_fixture() -> PartitionedFeatureDataset:
    return PartitionedFeatureDataset(
        feature_artifact_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        partitions=(
            partition_fixture(DatasetPartition.TRAIN, prefix="train"),
            partition_fixture(DatasetPartition.VALIDATION, prefix="validation"),
        ),
    )


def patch_loader(
    monkeypatch: pytest.MonkeyPatch,
    dataset: PartitionedFeatureDataset,
) -> list[tuple[Path, Path, str]]:
    calls: list[tuple[Path, Path, str]] = []

    def load(
        feature_artifact: str | Path,
        split_manifest: str | Path,
        *,
        expected_manifest_sha256: str,
    ) -> PartitionedFeatureDataset:
        calls.append((Path(feature_artifact), Path(split_manifest), expected_manifest_sha256))
        return dataset

    monkeypatch.setattr(report_module, "load_partitioned_feature_dataset", load)
    return calls


def small_training_config() -> PrototypeTrainingConfig:
    return PrototypeTrainingConfig(
        episode=EpisodeConfig(
            support_examples_per_class=1,
            query_examples=10,
            episodes=2,
        )
    )


def test_exports_deterministic_bundle_and_validation_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = dataset_fixture()
    calls = patch_loader(monkeypatch, dataset)
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    features.write_bytes(b"features")
    manifest.write_bytes(b"splits")
    first_bundle = tmp_path / "first" / "prototype-model.json"
    first_report = tmp_path / "first" / "prototype-validation.json"
    replay_bundle = tmp_path / "replay" / "prototype-model.json"
    replay_report = tmp_path / "replay" / "prototype-validation.json"

    first = export_prototype_validation_report(
        features,
        manifest,
        first_bundle,
        first_report,
        expected_manifest_sha256="b" * 64,
        training_config=small_training_config(),
        inference_config=PrototypeInferenceConfig(maximum_support_per_class=3),
    )
    replay = export_prototype_validation_report(
        features,
        manifest,
        replay_bundle,
        replay_report,
        expected_manifest_sha256="b" * 64,
        training_config=small_training_config(),
        inference_config=PrototypeInferenceConfig(maximum_support_per_class=3),
    )

    assert calls == [(features, manifest, "b" * 64)] * 2
    assert first_bundle.read_bytes() == replay_bundle.read_bytes()
    assert first_report.read_bytes() == replay_report.read_bytes()
    assert first.model_bundle_sha256 == sha256(first_bundle.read_bytes()).hexdigest()
    assert first.output_sha256 == sha256(first_report.read_bytes()).hexdigest()
    assert first.model_bundle_file_size_bytes == first_bundle.stat().st_size
    assert first.output_file_size_bytes == first_report.stat().st_size
    assert first.feature_artifact == str(features)
    assert first.feature_artifact_file_size_bytes == len(b"features")
    assert first.split_manifest == str(manifest)
    assert first.split_manifest_file_size_bytes == len(b"splits")
    assert first.validation.as_dict() == replay.validation.as_dict()

    report_payload = json.loads(first_report.read_text(encoding="utf-8"))
    bundle_payload = json.loads(first_bundle.read_text(encoding="utf-8"))
    assert report_payload == first.as_dict()
    assert report_payload["schema_version"] == PROTOTYPE_VALIDATION_REPORT_SCHEMA_VERSION
    assert report_payload["provenance"] == {
        "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
        "feature_artifact": {
            "file_name": features.name,
            "file_size_bytes": len(b"features"),
            "sha256": "a" * 64,
        },
        "split_manifest": {
            "file_name": manifest.name,
            "file_size_bytes": len(b"splits"),
            "sha256": "b" * 64,
        },
    }
    assert report_payload["model_bundle"] == {
        "file_name": first_bundle.name,
        "file_size_bytes": first_bundle.stat().st_size,
        "sha256": sha256(first_bundle.read_bytes()).hexdigest(),
        "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    }
    assert report_payload["evaluation_policy"] == {
        "fit_partition": "train",
        "evaluation_partition": "validation",
        "calibration_evaluated": False,
        "test_evaluated": False,
    }
    assert report_payload["validation_schema_version"] == PROTOTYPE_VALIDATION_SCHEMA_VERSION

    assert bundle_payload["schema_version"] == PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION
    assert bundle_payload["provenance"] == {
        "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
        "feature_artifact_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
    }
    model = bundle_payload["model"]
    assert isinstance(model, dict)
    assert model["architecture_schema_version"] == PROTOTYPE_MODEL_SCHEMA_VERSION
    assert model["training_schema_version"] == PROTOTYPE_TRAINING_SCHEMA_VERSION
    state_dict = model["state_dict"]
    assert isinstance(state_dict, dict)
    first_weight = state_dict["layers.0.weight"]
    assert isinstance(first_weight, dict)
    assert first_weight["shape"] == [64, FEATURE_COUNT]
    assert first_weight["dtype"] == "float32"
    assert len(first_weight["values"]) == 64
    inference = bundle_payload["inference"]
    assert isinstance(inference, dict)
    assert inference["support_examples_per_class"] == [3, 3, 3, 3, 3]
    class_means = inference["class_means"]
    class_variances = inference["class_variances"]
    assert isinstance(class_means, dict)
    assert isinstance(class_variances, dict)
    assert class_means["shape"] == [5, 64]
    assert class_variances["shape"] == [5, 64]


@pytest.mark.parametrize(
    ("bundle_name", "report_name", "existing", "message"),
    [
        ("model.txt", "report.json", None, "bundle must use the .json extension"),
        ("model.json", "report.txt", None, "report must use the .json extension"),
        ("same.json", "same.json", None, "must be different files"),
        ("model.json", "report.json", "bundle", "output already exists"),
        ("model.json", "report.json", "report", "output already exists"),
    ],
)
def test_rejects_invalid_destinations(
    tmp_path: Path,
    bundle_name: str,
    report_name: str,
    existing: str | None,
    message: str,
) -> None:
    bundle = tmp_path / bundle_name
    report = tmp_path / report_name
    if existing == "bundle":
        bundle.write_text("existing", encoding="utf-8")
    if existing == "report":
        report.write_text("existing", encoding="utf-8")

    with pytest.raises(PrototypeValidationReportError, match=message):
        export_prototype_validation_report(
            tmp_path / "features.parquet",
            tmp_path / "capture-splits.json",
            bundle,
            report,
            expected_manifest_sha256="b" * 64,
        )


def test_wraps_modeling_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise ModelingDatasetError("untrusted dataset")

    monkeypatch.setattr(report_module, "load_partitioned_feature_dataset", fail)

    with pytest.raises(
        PrototypeValidationReportError,
        match="experiment failed: untrusted dataset",
    ):
        export_prototype_validation_report(
            tmp_path / "features.parquet",
            tmp_path / "capture-splits.json",
            tmp_path / "model.json",
            tmp_path / "report.json",
            expected_manifest_sha256="b" * 64,
        )
