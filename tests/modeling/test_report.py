"""Tests for immutable provenance-bound baseline validation reports."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

import parallax.modeling.report as report_module
from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    BASELINE_EXPERIMENT_SCHEMA_VERSION,
    BASELINE_VALIDATION_REPORT_SCHEMA_VERSION,
    MODELING_DATASET_SCHEMA_VERSION,
    BaselineConfig,
    BaselineExperimentResult,
    BaselineValidationReportError,
    FeaturePartition,
    ModelingDatasetError,
    PartitionedFeatureDataset,
    export_baseline_validation_report,
    fit_initial_baselines,
)


def partition_fixture(
    partition: DatasetPartition,
    *,
    captures_prefix: str,
) -> FeaturePartition:
    categories = tuple(TrafficCategory)
    row_count = len(categories) * 2
    features = np.zeros((row_count, FEATURE_COUNT), dtype=np.float32)
    labels: list[str] = []
    captures: list[str] = []
    vpn_statuses: list[str] = []
    applications: list[str] = []

    row = 0
    for category_index, category in enumerate(categories):
        for sample_index in range(2):
            features[row, category_index] = 10.0 + sample_index
            labels.append(category.value)
            captures.append(f"{captures_prefix}-{category.value}-{sample_index}.pcap")
            vpn_statuses.append("vpn" if sample_index else "nonvpn")
            applications.append(f"application-{category_index}")
            row += 1

    vectors = (
        features,
        np.asarray(labels, dtype=np.str_),
        np.asarray(captures, dtype=np.str_),
        np.asarray(vpn_statuses, dtype=np.str_),
        np.asarray(applications, dtype=np.str_),
    )
    for vector in vectors:
        vector.setflags(write=False)

    return FeaturePartition(
        partition=partition,
        features=vectors[0],
        categories=vectors[1],
        capture_ids=vectors[2],
        vpn_statuses=vectors[3],
        applications=vectors[4],
    )


def dataset_fixture() -> PartitionedFeatureDataset:
    return PartitionedFeatureDataset(
        feature_artifact_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        partitions=(
            partition_fixture(DatasetPartition.TRAIN, captures_prefix="train"),
            partition_fixture(DatasetPartition.VALIDATION, captures_prefix="validation"),
        ),
    )


def patch_dataset_loader(
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


def test_exports_deterministic_validation_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = dataset_fixture()
    calls = patch_dataset_loader(monkeypatch, dataset)
    feature_artifact = tmp_path / "features.parquet"
    split_manifest = tmp_path / "capture-splits.json"
    feature_artifact.write_bytes(b"features")
    split_manifest.write_bytes(b"splits")
    first_output = tmp_path / "first" / "baseline-validation.json"
    replay_output = tmp_path / "replay" / "baseline-validation.json"

    first = export_baseline_validation_report(
        feature_artifact,
        split_manifest,
        first_output,
        expected_manifest_sha256="b" * 64,
    )
    replay = export_baseline_validation_report(
        feature_artifact,
        split_manifest,
        replay_output,
        expected_manifest_sha256="b" * 64,
    )

    assert calls == [
        (feature_artifact, split_manifest, "b" * 64),
        (feature_artifact, split_manifest, "b" * 64),
    ]
    assert first_output.read_bytes() == replay_output.read_bytes()
    assert first.output == str(first_output)
    assert first.output_file_size_bytes == first_output.stat().st_size
    assert first.output_sha256 == sha256(first_output.read_bytes()).hexdigest()
    assert first.feature_artifact == str(feature_artifact)
    assert first.feature_artifact_file_size_bytes == len(b"features")
    assert first.split_manifest == str(split_manifest)
    assert first.split_manifest_file_size_bytes == len(b"splits")
    assert first.logistic_iterations == replay.logistic_iterations
    assert min(first.logistic_iterations) > 0

    payload = json.loads(first_output.read_text(encoding="utf-8"))
    assert payload == first.as_dict()
    assert payload["schema_version"] == BASELINE_VALIDATION_REPORT_SCHEMA_VERSION
    assert payload["experiment_schema_version"] == BASELINE_EXPERIMENT_SCHEMA_VERSION
    assert payload["provenance"] == {
        "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
        "feature_artifact": {
            "file_name": feature_artifact.name,
            "file_size_bytes": len(b"features"),
            "sha256": "a" * 64,
        },
        "split_manifest": {
            "file_name": split_manifest.name,
            "file_size_bytes": len(b"splits"),
            "sha256": "b" * 64,
        },
    }
    assert payload["evaluation_policy"] == {
        "fit_partition": "train",
        "evaluation_partition": "validation",
        "calibration_evaluated": False,
        "test_evaluated": False,
    }
    assert payload["convergence"] == {
        "model": "balanced-logistic-regression",
        "iterations": list(first.logistic_iterations),
        "maximum_iterations": 2000,
        "converged_before_maximum_iterations": True,
    }
    experiment = payload["experiment"]
    assert isinstance(experiment, dict)
    assert experiment["data"] == {
        "training_windows": 10,
        "validation_windows": 10,
        "feature_count": FEATURE_COUNT,
        "evaluation_partition": "validation",
    }


@pytest.mark.parametrize(
    ("output_name", "existing", "message"),
    [
        ("baseline-validation.txt", False, "must use the .json extension"),
        ("baseline-validation.json", True, "already exists"),
    ],
)
def test_rejects_invalid_report_destination(
    tmp_path: Path,
    output_name: str,
    existing: bool,
    message: str,
) -> None:
    output = tmp_path / output_name
    if existing:
        output.write_text("existing", encoding="utf-8")

    with pytest.raises(BaselineValidationReportError, match=message):
        export_baseline_validation_report(
            tmp_path / "features.parquet",
            tmp_path / "capture-splits.json",
            output,
            expected_manifest_sha256="b" * 64,
        )


def test_wraps_modeling_dataset_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        feature_artifact: str | Path,
        split_manifest: str | Path,
        *,
        expected_manifest_sha256: str,
    ) -> NoReturn:
        raise ModelingDatasetError("untrusted dataset")

    monkeypatch.setattr(report_module, "load_partitioned_feature_dataset", fail)

    with pytest.raises(
        BaselineValidationReportError,
        match="experiment failed: untrusted dataset",
    ):
        export_baseline_validation_report(
            tmp_path / "features.parquet",
            tmp_path / "capture-splits.json",
            tmp_path / "baseline-validation.json",
            expected_manifest_sha256="b" * 64,
        )


def test_refuses_report_without_proven_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = dataset_fixture()
    patch_dataset_loader(monkeypatch, dataset)
    experiment = fit_initial_baselines(
        dataset.partition(DatasetPartition.TRAIN),
        dataset.partition(DatasetPartition.VALIDATION),
    )
    nonconverged = replace(
        experiment,
        config=BaselineConfig(maximum_iterations=1),
    )

    def fit(*args: object, **kwargs: object) -> BaselineExperimentResult:
        return nonconverged

    monkeypatch.setattr(report_module, "fit_initial_baselines", fit)

    with pytest.raises(
        BaselineValidationReportError,
        match="without proven convergence",
    ):
        export_baseline_validation_report(
            tmp_path / "features.parquet",
            tmp_path / "capture-splits.json",
            tmp_path / "baseline-validation.json",
            expected_manifest_sha256="b" * 64,
        )
