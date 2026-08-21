"""Immutable provenance-bound reports for initial validation baselines."""

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from parallax.data import DatasetPartition
from parallax.modeling.baselines import (
    BASELINE_EXPERIMENT_SCHEMA_VERSION,
    BaselineConfig,
    BaselineExperimentResult,
    BaselineModel,
    BaselineModelingError,
    fit_initial_baselines,
)
from parallax.modeling.dataset import (
    MODELING_DATASET_SCHEMA_VERSION,
    ModelingDatasetError,
    load_partitioned_feature_dataset,
)

BASELINE_VALIDATION_REPORT_SCHEMA_VERSION: Final = "vnat-baseline-validation-report-1"


class BaselineValidationReportError(ValueError):
    """Raised when a baseline validation report cannot be published safely."""


@dataclass(frozen=True, slots=True)
class BaselineValidationReport:
    """Published report metadata and its in-memory experiment result."""

    feature_artifact: str
    feature_artifact_file_size_bytes: int
    feature_artifact_sha256: str
    split_manifest: str
    split_manifest_file_size_bytes: int
    split_manifest_sha256: str
    output: str
    output_file_size_bytes: int
    output_sha256: str
    logistic_iterations: tuple[int, ...]
    experiment: BaselineExperimentResult

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation written to the report."""
        return _report_payload(
            feature_artifact_name=Path(self.feature_artifact).name,
            feature_artifact_file_size_bytes=self.feature_artifact_file_size_bytes,
            feature_artifact_sha256=self.feature_artifact_sha256,
            split_manifest_name=Path(self.split_manifest).name,
            split_manifest_file_size_bytes=self.split_manifest_file_size_bytes,
            split_manifest_sha256=self.split_manifest_sha256,
            logistic_iterations=self.logistic_iterations,
            experiment=self.experiment,
        )


def export_baseline_validation_report(
    feature_artifact: str | Path,
    split_manifest: str | Path,
    output: str | Path,
    *,
    expected_manifest_sha256: str,
    config: BaselineConfig | None = None,
) -> BaselineValidationReport:
    """Fit training-only baselines and atomically publish validation metrics."""
    feature_path = Path(feature_artifact)
    manifest_path = Path(split_manifest)
    destination = Path(output)
    _validate_destination(destination)

    try:
        dataset = load_partitioned_feature_dataset(
            feature_path,
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        experiment = fit_initial_baselines(
            dataset.partition(DatasetPartition.TRAIN),
            dataset.partition(DatasetPartition.VALIDATION),
            config=config,
        )
    except (ModelingDatasetError, BaselineModelingError, OSError) as error:
        raise BaselineValidationReportError(
            f"baseline validation experiment failed: {error}"
        ) from error

    logistic_iterations = _logistic_iterations(experiment)
    if max(logistic_iterations) >= experiment.config.maximum_iterations:
        raise BaselineValidationReportError(
            "refusing to publish a baseline report without proven convergence"
        )

    payload = _report_payload(
        feature_artifact_name=feature_path.name,
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest_name=manifest_path.name,
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        logistic_iterations=logistic_iterations,
        experiment=experiment,
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".parallax-baseline-report-", dir=destination.parent
    ) as temporary:
        temporary_output = Path(temporary) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(temporary_output, destination)

    return BaselineValidationReport(
        feature_artifact=str(feature_path),
        feature_artifact_file_size_bytes=feature_path.stat().st_size,
        feature_artifact_sha256=dataset.feature_artifact_sha256,
        split_manifest=str(manifest_path),
        split_manifest_file_size_bytes=manifest_path.stat().st_size,
        split_manifest_sha256=dataset.split_manifest_sha256,
        output=str(destination),
        output_file_size_bytes=len(encoded),
        output_sha256=sha256(encoded).hexdigest(),
        logistic_iterations=logistic_iterations,
        experiment=experiment,
    )


def _validate_destination(destination: Path) -> None:
    if destination.suffix.casefold() != ".json":
        raise BaselineValidationReportError("baseline report output must use the .json extension")
    if destination.exists():
        raise BaselineValidationReportError(f"baseline report already exists: {destination}")


def _logistic_iterations(experiment: BaselineExperimentResult) -> tuple[int, ...]:
    estimator = experiment.model(BaselineModel.BALANCED_LOGISTIC_REGRESSION).estimator
    classifier = cast("LogisticRegression", estimator.named_steps["classifier"])
    return tuple(int(value) for value in np.atleast_1d(classifier.n_iter_))


def _report_payload(
    *,
    feature_artifact_name: str,
    feature_artifact_file_size_bytes: int,
    feature_artifact_sha256: str,
    split_manifest_name: str,
    split_manifest_file_size_bytes: int,
    split_manifest_sha256: str,
    logistic_iterations: tuple[int, ...],
    experiment: BaselineExperimentResult,
) -> dict[str, object]:
    return {
        "schema_version": BASELINE_VALIDATION_REPORT_SCHEMA_VERSION,
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
        "evaluation_policy": {
            "fit_partition": DatasetPartition.TRAIN.value,
            "evaluation_partition": DatasetPartition.VALIDATION.value,
            "calibration_evaluated": False,
            "test_evaluated": False,
        },
        "convergence": {
            "model": BaselineModel.BALANCED_LOGISTIC_REGRESSION.value,
            "iterations": list(logistic_iterations),
            "maximum_iterations": experiment.config.maximum_iterations,
            "converged_before_maximum_iterations": True,
        },
        "experiment_schema_version": BASELINE_EXPERIMENT_SCHEMA_VERSION,
        "experiment": experiment.as_dict(),
    }
