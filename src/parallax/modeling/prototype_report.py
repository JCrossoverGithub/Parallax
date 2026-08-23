"""Immutable prototype model bundles and validation-only reports."""

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
from parallax.modeling.dataset import (
    MODELING_DATASET_SCHEMA_VERSION,
    ModelingDatasetError,
    load_partitioned_feature_dataset,
)
from parallax.modeling.inference import (
    PROTOTYPE_VALIDATION_SCHEMA_VERSION,
    PrototypeInferenceConfig,
    PrototypeInferenceError,
    PrototypeValidationResult,
    evaluate_prototype_validation,
)
from parallax.modeling.prototypes import PROTOTYPE_MODEL_SCHEMA_VERSION
from parallax.modeling.training import (
    PROTOTYPE_TRAINING_SCHEMA_VERSION,
    PrototypeTrainingConfig,
    PrototypeTrainingError,
    fit_prototype_embedding,
)

PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION: Final = "vnat-prototype-model-bundle-1"
PROTOTYPE_VALIDATION_REPORT_SCHEMA_VERSION: Final = "vnat-prototype-validation-report-1"


class PrototypeValidationReportError(ValueError):
    """Raised when a prototype bundle/report pair cannot be published safely."""


@dataclass(frozen=True, slots=True)
class PrototypeValidationReport:
    """Published artifact metadata and its in-memory validation result."""

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
    validation: PrototypeValidationResult

    def as_dict(self) -> dict[str, object]:
        """Return the stable compact report representation."""
        return _report_payload(
            feature_artifact_name=Path(self.feature_artifact).name,
            feature_artifact_file_size_bytes=self.feature_artifact_file_size_bytes,
            feature_artifact_sha256=self.feature_artifact_sha256,
            split_manifest_name=Path(self.split_manifest).name,
            split_manifest_file_size_bytes=self.split_manifest_file_size_bytes,
            split_manifest_sha256=self.split_manifest_sha256,
            model_bundle_name=Path(self.model_bundle).name,
            model_bundle_file_size_bytes=self.model_bundle_file_size_bytes,
            model_bundle_sha256=self.model_bundle_sha256,
            validation=self.validation,
        )


def export_prototype_validation_report(
    feature_artifact: str | Path,
    split_manifest: str | Path,
    model_bundle: str | Path,
    output: str | Path,
    *,
    expected_manifest_sha256: str,
    training_config: PrototypeTrainingConfig | None = None,
    inference_config: PrototypeInferenceConfig | None = None,
) -> PrototypeValidationReport:
    """Train, validate, and publish a deterministic bundle/report pair."""
    feature_path = Path(feature_artifact)
    manifest_path = Path(split_manifest)
    bundle_path = Path(model_bundle)
    report_path = Path(output)
    _validate_destinations(bundle_path, report_path)

    try:
        dataset = load_partitioned_feature_dataset(
            feature_path,
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        training = dataset.partition(DatasetPartition.TRAIN)
        validation_partition = dataset.partition(DatasetPartition.VALIDATION)
        training_result = fit_prototype_embedding(training, config=training_config)
        validation = evaluate_prototype_validation(
            training_result,
            training,
            validation_partition,
            config=inference_config,
        )
    except (
        ModelingDatasetError,
        PrototypeTrainingError,
        PrototypeInferenceError,
        OSError,
        RuntimeError,
    ) as error:
        raise PrototypeValidationReportError(
            f"prototype validation experiment failed: {error}"
        ) from error

    bundle_payload = _bundle_payload(
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest_sha256=dataset.split_manifest_sha256,
        validation=validation,
    )
    bundle_encoded = _encode(bundle_payload)
    bundle_sha256 = sha256(bundle_encoded).hexdigest()
    report_payload = _report_payload(
        feature_artifact_name=feature_path.name,
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest_name=manifest_path.name,
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle_name=bundle_path.name,
        model_bundle_file_size_bytes=len(bundle_encoded),
        model_bundle_sha256=bundle_sha256,
        validation=validation,
    )
    report_encoded = _encode(report_payload)

    _atomic_write(bundle_path, bundle_encoded, prefix=".parallax-prototype-bundle-")
    _atomic_write(report_path, report_encoded, prefix=".parallax-prototype-report-")

    return PrototypeValidationReport(
        feature_artifact=str(feature_path),
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest=str(manifest_path),
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle=str(bundle_path),
        model_bundle_file_size_bytes=len(bundle_encoded),
        model_bundle_sha256=bundle_sha256,
        output=str(report_path),
        output_file_size_bytes=len(report_encoded),
        output_sha256=sha256(report_encoded).hexdigest(),
        validation=validation,
    )


def _validate_destinations(bundle: Path, report: Path) -> None:
    if bundle.suffix.casefold() != ".json":
        raise PrototypeValidationReportError("prototype model bundle must use the .json extension")
    if report.suffix.casefold() != ".json":
        raise PrototypeValidationReportError(
            "prototype validation report must use the .json extension"
        )
    if bundle.resolve() == report.resolve():
        raise PrototypeValidationReportError(
            "model bundle and validation report must be different files"
        )
    for destination in (bundle, report):
        if destination.exists():
            raise PrototypeValidationReportError(f"prototype output already exists: {destination}")


def _bundle_payload(
    *,
    feature_artifact_sha256: str,
    split_manifest_sha256: str,
    validation: PrototypeValidationResult,
) -> dict[str, object]:
    training = validation.training
    state = validation.inference_state
    return {
        "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
        "provenance": {
            "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
            "feature_artifact_sha256": feature_artifact_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        },
        "model": {
            "architecture_schema_version": PROTOTYPE_MODEL_SCHEMA_VERSION,
            "training_schema_version": PROTOTYPE_TRAINING_SCHEMA_VERSION,
            "category_order": list(CATEGORY_LABELS),
            "state_dict": {
                name: _tensor_payload(tensor)
                for name, tensor in training.network.state_dict().items()
            },
        },
        "preprocessing": {
            "feature_mean": training.feature_mean.tolist(),
            "feature_scale": training.feature_scale.tolist(),
        },
        "inference": {
            "validation_schema_version": PROTOTYPE_VALIDATION_SCHEMA_VERSION,
            "configuration": validation.config.as_dict(),
            "support_indices": state.support_indices.tolist(),
            "support_examples_per_class": list(state.support_examples_per_class),
            "class_means": _tensor_payload(state.class_means),
            "class_variances": _tensor_payload(state.class_variances),
        },
    }


def _tensor_payload(tensor: Tensor) -> dict[str, object]:
    values = tensor.detach().cpu()
    return {
        "dtype": str(values.dtype).removeprefix("torch."),
        "shape": list(values.shape),
        "values": values.tolist(),
    }


def _report_payload(
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
    validation: PrototypeValidationResult,
) -> dict[str, object]:
    return {
        "schema_version": PROTOTYPE_VALIDATION_REPORT_SCHEMA_VERSION,
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
        },
        "model_bundle": {
            "file_name": model_bundle_name,
            "file_size_bytes": model_bundle_file_size_bytes,
            "sha256": model_bundle_sha256,
            "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
        },
        "evaluation_policy": {
            "fit_partition": DatasetPartition.TRAIN.value,
            "evaluation_partition": DatasetPartition.VALIDATION.value,
            "calibration_evaluated": False,
            "test_evaluated": False,
        },
        "validation_schema_version": PROTOTYPE_VALIDATION_SCHEMA_VERSION,
        "validation": validation.as_dict(),
    }


def _encode(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(destination: Path, encoded: bytes, *, prefix: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=prefix, dir=destination.parent) as temporary:
        temporary_output = Path(temporary) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(temporary_output, destination)
