"""Immutable export of frozen-model OOD calibration artifacts."""

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from torch import Tensor

from parallax.data import DatasetPartition
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.bundle import PrototypeBundleError, load_prototype_model_bundle
from parallax.modeling.calibration import (
    PrototypeCalibrationError,
    PrototypeCalibrationResult,
    calibrate_prototype_ood,
)
from parallax.modeling.dataset import (
    MODELING_DATASET_SCHEMA_VERSION,
    ModelingDatasetError,
    load_partitioned_feature_dataset,
)
from parallax.modeling.prototype_report import PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION
from parallax.modeling.uncertainty import OOD_CALIBRATION_SCHEMA_VERSION

PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION: Final = "vnat-prototype-ood-calibration-artifact-1"


class PrototypeCalibrationArtifactError(ValueError):
    """Raised when a calibration artifact cannot be published safely."""


@dataclass(frozen=True, slots=True)
class PrototypeCalibrationArtifact:
    """Published calibration artifact metadata and in-memory fitted state."""

    feature_artifact: str
    feature_artifact_file_size_bytes: int
    feature_artifact_sha256: str
    split_manifest: str
    split_manifest_file_size_bytes: int
    split_manifest_sha256: str
    model_bundle: str
    model_bundle_file_size_bytes: int
    model_bundle_sha256: str
    output: str
    output_file_size_bytes: int
    output_sha256: str
    calibration: PrototypeCalibrationResult

    def as_dict(self) -> dict[str, object]:
        """Return the deterministic calibration artifact payload."""
        return _artifact_payload(
            feature_artifact_name=Path(self.feature_artifact).name,
            feature_artifact_file_size_bytes=self.feature_artifact_file_size_bytes,
            feature_artifact_sha256=self.feature_artifact_sha256,
            split_manifest_name=Path(self.split_manifest).name,
            split_manifest_file_size_bytes=self.split_manifest_file_size_bytes,
            split_manifest_sha256=self.split_manifest_sha256,
            model_bundle_name=Path(self.model_bundle).name,
            model_bundle_file_size_bytes=self.model_bundle_file_size_bytes,
            model_bundle_sha256=self.model_bundle_sha256,
            calibration=self.calibration,
        )


def export_prototype_ood_calibration(
    feature_artifact: str | Path,
    split_manifest: str | Path,
    model_bundle: str | Path,
    output: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_model_bundle_sha256: str,
) -> PrototypeCalibrationArtifact:
    """Load a frozen model and publish training/calibration-derived OOD state."""
    feature_path = Path(feature_artifact)
    manifest_path = Path(split_manifest)
    bundle_path = Path(model_bundle)
    destination = Path(output)
    _validate_destination(destination)
    try:
        dataset = load_partitioned_feature_dataset(
            feature_path,
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        bundle = load_prototype_model_bundle(
            bundle_path,
            expected_sha256=expected_model_bundle_sha256,
            expected_feature_artifact_sha256=dataset.feature_artifact_sha256,
            expected_split_manifest_sha256=dataset.split_manifest_sha256,
        )
        calibration = calibrate_prototype_ood(
            bundle,
            dataset.partition(DatasetPartition.TRAIN),
            dataset.partition(DatasetPartition.CALIBRATION),
        )
    except (
        ModelingDatasetError,
        PrototypeBundleError,
        PrototypeCalibrationError,
        OSError,
        RuntimeError,
    ) as error:
        raise PrototypeCalibrationArtifactError(
            f"prototype OOD calibration export failed: {error}"
        ) from error

    payload = _artifact_payload(
        feature_artifact_name=feature_path.name,
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest_name=manifest_path.name,
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle_name=bundle_path.name,
        model_bundle_file_size_bytes=bundle_path.stat().st_size,
        model_bundle_sha256=bundle.sha256,
        calibration=calibration,
    )
    encoded = _encode(payload)
    _atomic_write(destination, encoded)
    return PrototypeCalibrationArtifact(
        feature_artifact=str(feature_path),
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest=str(manifest_path),
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle=str(bundle_path),
        model_bundle_file_size_bytes=bundle_path.stat().st_size,
        model_bundle_sha256=bundle.sha256,
        output=str(destination),
        output_file_size_bytes=len(encoded),
        output_sha256=sha256(encoded).hexdigest(),
        calibration=calibration,
    )


def _validate_destination(destination: Path) -> None:
    if destination.suffix.casefold() != ".json":
        raise PrototypeCalibrationArtifactError("calibration output must use the .json extension")
    if destination.exists():
        raise PrototypeCalibrationArtifactError(f"calibration output already exists: {destination}")


def _artifact_payload(
    *,
    feature_artifact_name: str,
    feature_artifact_file_size_bytes: int,
    feature_artifact_sha256: str,
    split_manifest_name: str,
    split_manifest_file_size_bytes: int,
    split_manifest_sha256: str,
    model_bundle_name: str,
    model_bundle_file_size_bytes: int,
    model_bundle_sha256: str,
    calibration: PrototypeCalibrationResult,
) -> dict[str, object]:
    geometry = calibration.geometry
    return {
        "schema_version": PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
            "feature_artifact": {
                "file_name": feature_artifact_name,
                "file_size_bytes": feature_artifact_file_size_bytes,
                "sha256": feature_artifact_sha256,
            },
            "split_manifest": {
                "file_name": split_manifest_name,
                "file_size_bytes": split_manifest_file_size_bytes,
                "sha256": split_manifest_sha256,
            },
            "model_bundle": {
                "file_name": model_bundle_name,
                "file_size_bytes": model_bundle_file_size_bytes,
                "sha256": model_bundle_sha256,
                "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
            },
        },
        "evaluation_policy": {
            "geometry_fit_partition": DatasetPartition.TRAIN.value,
            "density_fit_partition": DatasetPartition.CALIBRATION.value,
            "model_selection_performed": False,
            "validation_evaluated": False,
            "test_evaluated": False,
        },
        "calibration_schema_version": OOD_CALIBRATION_SCHEMA_VERSION,
        "calibration": calibration.as_dict(),
        "geometry": {
            "class_means": _tensor_payload(geometry.class_means),
            "class_covariances": _tensor_payload(geometry.class_covariances),
            "global_mean": _tensor_payload(geometry.global_mean),
            "global_covariance": _tensor_payload(geometry.global_covariance),
        },
        "class_kdes": {
            category: {
                **kde.as_dict(),
                "relative_distance_samples": kde.samples.tolist(),
            }
            for category, kde in zip(CATEGORY_LABELS, calibration.class_kdes, strict=True)
        },
    }


def _tensor_payload(tensor: Tensor) -> dict[str, object]:
    values = tensor.detach().cpu()
    return {
        "dtype": str(values.dtype).removeprefix("torch."),
        "shape": list(values.shape),
        "values": values.tolist(),
    }


def _encode(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(destination: Path, encoded: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parallax-ood-calibration-", dir=destination.parent) as temp:
        temporary_output = Path(temp) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(temporary_output, destination)
