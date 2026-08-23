"""Tests for immutable prototype OOD calibration artifacts."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from parallax.data import DatasetPartition
from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    OOD_CALIBRATION_SCHEMA_VERSION,
    PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    FeaturePartition,
    GaussianKDE1D,
    LoadedPrototypeBundle,
    PartitionedFeatureDataset,
    PrototypeCalibrationArtifactError,
    PrototypeCalibrationResult,
    PrototypeEmbeddingNetwork,
    PrototypeInferenceConfig,
    PrototypeInferenceState,
    RelativeMahalanobisState,
    export_prototype_ood_calibration,
)


def empty_partition(partition: DatasetPartition) -> FeaturePartition:
    features = np.zeros((1, 129), dtype=np.float32)
    strings = np.asarray(["value"], dtype=np.str_)
    features.setflags(write=False)
    strings.setflags(write=False)
    return FeaturePartition(
        partition=partition,
        features=features,
        categories=strings,
        capture_ids=strings,
        vpn_statuses=strings,
        applications=strings,
    )


def calibration_fixture() -> PrototypeCalibrationResult:
    network = PrototypeEmbeddingNetwork()
    mean = np.zeros(129, dtype=np.float64)
    scale = np.ones(129, dtype=np.float64)
    mean.setflags(write=False)
    scale.setflags(write=False)
    bundle = LoadedPrototypeBundle(
        sha256="c" * 64,
        feature_artifact_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        network=network,
        feature_mean=mean,
        feature_scale=scale,
        inference_config=PrototypeInferenceConfig(maximum_support_per_class=2),
        inference_state=PrototypeInferenceState(
            support_indices=torch.arange(10),
            support_examples_per_class=(2, 2, 2, 2, 2),
            class_means=torch.zeros((5, EMBEDDING_DIMENSION)),
            class_variances=torch.ones((5, EMBEDDING_DIMENSION)),
        ),
    )
    geometry = RelativeMahalanobisState(
        class_means=torch.zeros((5, EMBEDDING_DIMENSION)),
        class_covariances=torch.stack([torch.eye(EMBEDDING_DIMENSION)] * 5),
        global_mean=torch.zeros(EMBEDDING_DIMENSION),
        global_covariance=torch.eye(EMBEDDING_DIMENSION),
    )
    kdes = tuple(
        GaussianKDE1D(
            samples=np.asarray([float(index), float(index + 1)], dtype=np.float64),
            bandwidth=0.5,
        )
        for index in range(5)
    )
    scores = np.concatenate([kde.samples for kde in kdes])
    scores.setflags(write=False)
    return PrototypeCalibrationResult(
        bundle=bundle,
        geometry=geometry,
        class_kdes=kdes,
        relative_scores=scores,
        calibration_examples_per_class=(2, 2, 2, 2, 2),
        calibration_windows=10,
    )


def test_exports_replayable_provenance_bound_calibration_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    bundle_path = tmp_path / "prototype-model.json"
    for path, content in (
        (features, b"features"),
        (manifest, b"manifest"),
        (bundle_path, b"bundle"),
    ):
        path.write_bytes(content)
    calibration = calibration_fixture()
    dataset = PartitionedFeatureDataset(
        feature_artifact_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        partitions=(
            empty_partition(DatasetPartition.TRAIN),
            empty_partition(DatasetPartition.CALIBRATION),
        ),
    )
    calls: list[object] = []

    def fake_dataset(
        selected_features: Path,
        selected_manifest: Path,
        *,
        expected_manifest_sha256: str,
    ) -> PartitionedFeatureDataset:
        calls.append((selected_features, selected_manifest, expected_manifest_sha256))
        return dataset

    def fake_bundle(
        selected_bundle: Path,
        *,
        expected_sha256: str,
        expected_feature_artifact_sha256: str,
        expected_split_manifest_sha256: str,
    ) -> LoadedPrototypeBundle:
        calls.append(
            (
                selected_bundle,
                expected_sha256,
                expected_feature_artifact_sha256,
                expected_split_manifest_sha256,
            )
        )
        return calibration.bundle

    def fake_calibrate(
        selected_bundle: LoadedPrototypeBundle,
        training: FeaturePartition,
        selected_calibration: FeaturePartition,
    ) -> PrototypeCalibrationResult:
        calls.append((selected_bundle, training.partition, selected_calibration.partition))
        return calibration

    monkeypatch.setattr(
        "parallax.modeling.calibration_report.load_partitioned_feature_dataset",
        fake_dataset,
    )
    monkeypatch.setattr(
        "parallax.modeling.calibration_report.load_prototype_model_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "parallax.modeling.calibration_report.calibrate_prototype_ood",
        fake_calibrate,
    )
    first_path = tmp_path / "first" / "prototype-calibration.json"
    replay_path = tmp_path / "replay" / "prototype-calibration.json"

    first = export_prototype_ood_calibration(
        features,
        manifest,
        bundle_path,
        first_path,
        expected_manifest_sha256="b" * 64,
        expected_model_bundle_sha256="c" * 64,
    )
    replay = export_prototype_ood_calibration(
        features,
        manifest,
        bundle_path,
        replay_path,
        expected_manifest_sha256="b" * 64,
        expected_model_bundle_sha256="c" * 64,
    )

    assert first_path.read_bytes() == replay_path.read_bytes()
    assert first.as_dict() == replay.as_dict()
    assert first.output_sha256 == sha256(first_path.read_bytes()).hexdigest()
    assert first.output_file_size_bytes == first_path.stat().st_size
    assert first.feature_artifact == str(features)
    assert first.split_manifest == str(manifest)
    assert first.model_bundle == str(bundle_path)
    assert len(calls) == 6

    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION
    assert payload["calibration_schema_version"] == OOD_CALIBRATION_SCHEMA_VERSION
    assert payload["provenance"]["model_bundle"] == {
        "file_name": bundle_path.name,
        "file_size_bytes": len(b"bundle"),
        "sha256": "c" * 64,
        "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    }
    assert payload["evaluation_policy"] == {
        "geometry_fit_partition": "train",
        "density_fit_partition": "calibration",
        "model_selection_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
    }
    assert payload["geometry"]["class_covariances"]["shape"] == [5, 64, 64]
    assert payload["geometry"]["global_covariance"]["shape"] == [64, 64]
    assert set(payload["class_kdes"]) == set(CATEGORY_LABELS)
    assert payload["class_kdes"]["STREAMING"] == {
        "bandwidth": 0.5,
        "bandwidth_method": "scott",
        "relative_distance_samples": [0.0, 1.0],
        "sample_count": 2,
    }


@pytest.mark.parametrize(
    ("name", "existing", "message"),
    [
        ("calibration.txt", False, "must use the .json extension"),
        ("calibration.json", True, "already exists"),
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

    with pytest.raises(PrototypeCalibrationArtifactError, match=message):
        export_prototype_ood_calibration(
            tmp_path / "features.parquet",
            tmp_path / "manifest.json",
            tmp_path / "model.json",
            output,
            expected_manifest_sha256="b" * 64,
            expected_model_bundle_sha256="c" * 64,
        )


def test_wraps_calibration_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr(
        "parallax.modeling.calibration_report.load_partitioned_feature_dataset",
        fail,
    )

    with pytest.raises(PrototypeCalibrationArtifactError, match="export failed: unavailable"):
        export_prototype_ood_calibration(
            tmp_path / "features.parquet",
            tmp_path / "manifest.json",
            tmp_path / "model.json",
            tmp_path / "calibration.json",
            expected_manifest_sha256="b" * 64,
            expected_model_bundle_sha256="c" * 64,
        )
