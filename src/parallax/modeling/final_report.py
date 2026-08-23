"""Immutable one-shot test reports for frozen calibrated prototype models."""

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from parallax.data import DatasetPartition
from parallax.modeling.bundle import PrototypeBundleError, load_prototype_model_bundle
from parallax.modeling.calibration_bundle import (
    PrototypeCalibrationBundleError,
    load_prototype_ood_calibration,
)
from parallax.modeling.calibration_report import (
    PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
)
from parallax.modeling.dataset import (
    MODELING_DATASET_SCHEMA_VERSION,
    ModelingDatasetError,
    load_partitioned_feature_dataset,
)
from parallax.modeling.evaluation import (
    PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION,
    PrototypeEvaluationError,
    PrototypeTestEvaluation,
    evaluate_prototype_test,
)
from parallax.modeling.prototype_report import PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION

PROTOTYPE_TEST_REPORT_SCHEMA_VERSION: Final = "vnat-prototype-test-report-1"


class PrototypeTestReportError(ValueError):
    """Raised when final test evidence cannot be published safely."""


@dataclass(frozen=True, slots=True)
class PrototypeTestReport:
    """Published report metadata and its one-shot test evaluation."""

    feature_artifact: str
    feature_artifact_file_size_bytes: int
    feature_artifact_sha256: str
    split_manifest: str
    split_manifest_file_size_bytes: int
    split_manifest_sha256: str
    model_bundle: str
    model_bundle_file_size_bytes: int
    model_bundle_sha256: str
    calibration_artifact: str
    calibration_artifact_file_size_bytes: int
    calibration_artifact_sha256: str
    output: str
    output_file_size_bytes: int
    output_sha256: str
    evaluation: PrototypeTestEvaluation

    def as_dict(self) -> dict[str, object]:
        """Return the deterministic final test report payload."""
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
            calibration_artifact_name=Path(self.calibration_artifact).name,
            calibration_artifact_file_size_bytes=self.calibration_artifact_file_size_bytes,
            calibration_artifact_sha256=self.calibration_artifact_sha256,
            evaluation=self.evaluation,
        )


def export_prototype_test_report(
    feature_artifact: str | Path,
    split_manifest: str | Path,
    model_bundle: str | Path,
    calibration_artifact: str | Path,
    output: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_model_bundle_sha256: str,
    expected_calibration_artifact_sha256: str,
) -> PrototypeTestReport:
    """Evaluate the frozen candidate once and atomically publish test evidence."""
    feature_path = Path(feature_artifact)
    manifest_path = Path(split_manifest)
    bundle_path = Path(model_bundle)
    calibration_path = Path(calibration_artifact)
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
        calibration = load_prototype_ood_calibration(
            calibration_path,
            expected_sha256=expected_calibration_artifact_sha256,
            expected_model_bundle_sha256=bundle.sha256,
            expected_feature_artifact_sha256=dataset.feature_artifact_sha256,
            expected_split_manifest_sha256=dataset.split_manifest_sha256,
        )
        evaluation = evaluate_prototype_test(
            bundle,
            calibration,
            dataset.partition(DatasetPartition.TEST),
        )
    except (
        ModelingDatasetError,
        PrototypeBundleError,
        PrototypeCalibrationBundleError,
        PrototypeEvaluationError,
        OSError,
        RuntimeError,
    ) as error:
        raise PrototypeTestReportError(f"prototype test report failed: {error}") from error

    payload = _report_payload(
        feature_artifact_name=feature_path.name,
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest_name=manifest_path.name,
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle_name=bundle_path.name,
        model_bundle_file_size_bytes=bundle_path.stat().st_size,
        model_bundle_sha256=bundle.sha256,
        calibration_artifact_name=calibration_path.name,
        calibration_artifact_file_size_bytes=calibration_path.stat().st_size,
        calibration_artifact_sha256=calibration.sha256,
        evaluation=evaluation,
    )
    encoded = _encode(payload)
    _atomic_write(destination, encoded)
    return PrototypeTestReport(
        feature_artifact=str(feature_path),
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest=str(manifest_path),
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        model_bundle=str(bundle_path),
        model_bundle_file_size_bytes=bundle_path.stat().st_size,
        model_bundle_sha256=bundle.sha256,
        calibration_artifact=str(calibration_path),
        calibration_artifact_file_size_bytes=calibration_path.stat().st_size,
        calibration_artifact_sha256=calibration.sha256,
        output=str(destination),
        output_file_size_bytes=len(encoded),
        output_sha256=sha256(encoded).hexdigest(),
        evaluation=evaluation,
    )


def _validate_destination(destination: Path) -> None:
    if destination.suffix.casefold() != ".json":
        raise PrototypeTestReportError("prototype test report must use the .json extension")
    if destination.exists():
        raise PrototypeTestReportError(f"prototype test report already exists: {destination}")


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
    calibration_artifact_name: str,
    calibration_artifact_file_size_bytes: int,
    calibration_artifact_sha256: str,
    evaluation: PrototypeTestEvaluation,
) -> dict[str, object]:
    return {
        "schema_version": PROTOTYPE_TEST_REPORT_SCHEMA_VERSION,
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
            "calibration_artifact": {
                "file_name": calibration_artifact_name,
                "file_size_bytes": calibration_artifact_file_size_bytes,
                "sha256": calibration_artifact_sha256,
                "schema_version": PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            },
        },
        "evaluation_policy": {
            "candidate_selection_partition": DatasetPartition.VALIDATION.value,
            "geometry_fit_partition": DatasetPartition.TRAIN.value,
            "density_fit_partition": DatasetPartition.CALIBRATION.value,
            "evaluation_partition": DatasetPartition.TEST.value,
            "candidate_frozen_before_test": True,
            "model_selection_after_test": False,
            "test_evaluated": True,
        },
        "test_evaluation_schema_version": PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION,
        "test": evaluation.as_dict(),
    }


def _encode(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(destination: Path, encoded: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parallax-prototype-test-", dir=destination.parent) as temp:
        temporary_output = Path(temp) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(temporary_output, destination)
