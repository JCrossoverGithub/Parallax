"""Tests for training-only majority and logistic-regression baselines."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    BASELINE_EXPERIMENT_SCHEMA_VERSION,
    CATEGORY_LABELS,
    BaselineConfig,
    BaselineModel,
    BaselineModelingError,
    FeaturePartition,
    fit_initial_baselines,
)


def partition_fixture(
    partition: DatasetPartition,
    *,
    captures_prefix: str,
    samples_per_category: int = 2,
) -> FeaturePartition:
    categories = tuple(TrafficCategory)
    row_count = len(categories) * samples_per_category
    features = np.zeros((row_count, FEATURE_COUNT), dtype=np.float32)
    labels: list[str] = []
    captures: list[str] = []
    vpn_statuses: list[str] = []
    applications: list[str] = []

    row = 0
    for category_index, category in enumerate(categories):
        for sample_index in range(samples_per_category):
            features[row, category_index] = 10.0 + sample_index
            labels.append(category.value)
            captures.append(f"{captures_prefix}-{category.value}-{sample_index}.pcap")
            vpn_statuses.append("vpn" if sample_index % 2 else "nonvpn")
            applications.append(f"application-{category_index}")
            row += 1

    category_array = np.asarray(labels, dtype=np.str_)
    capture_array = np.asarray(captures, dtype=np.str_)
    vpn_status_array = np.asarray(vpn_statuses, dtype=np.str_)
    application_array = np.asarray(applications, dtype=np.str_)
    for vector in (
        features,
        category_array,
        capture_array,
        vpn_status_array,
        application_array,
    ):
        vector.setflags(write=False)

    return FeaturePartition(
        partition=partition,
        features=features,
        categories=category_array,
        capture_ids=capture_array,
        vpn_statuses=vpn_status_array,
        applications=application_array,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"logistic_c": 0.0}, "C must be finite and positive"),
        ({"logistic_c": float("inf")}, "C must be finite and positive"),
        ({"maximum_iterations": 0}, "iterations must be positive"),
        ({"random_seed": -1}, "seed must not be negative"),
    ],
)
def test_rejects_invalid_baseline_configuration(
    changes: dict[str, float | int], message: str
) -> None:
    with pytest.raises(BaselineModelingError, match=message):
        BaselineConfig(**changes)  # type: ignore[arg-type]


def test_fits_deterministic_training_only_baselines() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, captures_prefix="train")
    validation = partition_fixture(DatasetPartition.VALIDATION, captures_prefix="validation")

    first = fit_initial_baselines(training, validation)
    replay = fit_initial_baselines(training, validation)

    assert first.as_dict() == replay.as_dict()
    assert first.training_windows == 10
    assert first.validation_windows == 10
    assert first.feature_count == FEATURE_COUNT
    assert tuple(model.model for model in first.models) == tuple(BaselineModel)

    payload = first.as_dict()
    assert payload["schema_version"] == BASELINE_EXPERIMENT_SCHEMA_VERSION
    assert payload["category_order"] == list(CATEGORY_LABELS)
    assert payload["configuration"] == {
        "logistic_c": 1.0,
        "maximum_iterations": 2000,
        "random_seed": 17,
        "logistic_class_weight": "balanced",
        "logistic_solver": "lbfgs",
        "scaler_fit_partition": "train",
    }
    assert payload["data"] == {
        "training_windows": 10,
        "validation_windows": 10,
        "feature_count": 129,
        "evaluation_partition": "validation",
    }

    majority = first.model(BaselineModel.MAJORITY_CLASS)
    majority_metrics = majority.validation_metrics
    assert majority_metrics.accuracy == pytest.approx(0.2)
    assert majority_metrics.balanced_accuracy == pytest.approx(0.2)
    assert majority_metrics.micro_f1 == pytest.approx(0.2)
    assert majority_metrics.macro_f1 == pytest.approx(1.0 / 15.0)
    assert sum(sum(row) for row in majority_metrics.confusion_matrix) == 10
    assert list(majority_metrics.categories) == list(CATEGORY_LABELS)
    assert all(metrics.support == 2 for metrics in majority_metrics.categories.values())

    logistic = first.model(BaselineModel.BALANCED_LOGISTIC_REGRESSION)
    logistic_metrics = logistic.validation_metrics
    assert logistic_metrics.accuracy == pytest.approx(1.0)
    assert logistic_metrics.balanced_accuracy == pytest.approx(1.0)
    assert logistic_metrics.micro_f1 == pytest.approx(1.0)
    assert logistic_metrics.macro_f1 == pytest.approx(1.0)
    assert logistic_metrics.confusion_matrix == tuple(
        tuple(int(row == column) * 2 for column in range(5)) for row in range(5)
    )

    scaler = logistic.estimator.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None
    assert np.allclose(
        scaler.mean_,
        np.mean(training.features, axis=0, dtype=np.float64),
    )

    with pytest.raises(KeyError, match="majority-class"):
        first.model("majority-class")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("training_change", "validation_change", "message"),
    [
        (
            {"partition": DatasetPartition.CALIBRATION},
            {},
            "expected train partition",
        ),
        (
            {},
            {"partition": DatasetPartition.TEST},
            "expected validation partition",
        ),
        (
            {"features": np.empty((0, FEATURE_COUNT), dtype=np.float32)},
            {},
            "nonempty and two-dimensional",
        ),
        (
            {"features": np.zeros((10, FEATURE_COUNT - 1), dtype=np.float32)},
            {},
            "must contain 129 columns",
        ),
        (
            {"features": np.zeros((10, FEATURE_COUNT), dtype=np.float64)},
            {},
            "must use float32",
        ),
        (
            {
                "features": np.full(
                    (10, FEATURE_COUNT),
                    np.nan,
                    dtype=np.float32,
                )
            },
            {},
            "only finite",
        ),
        (
            {"categories": np.asarray(["C2"], dtype=np.str_)},
            {},
            "vectors must align",
        ),
        (
            {},
            {"categories": np.asarray(["C2"] * 10, dtype=np.str_)},
            "every traffic category",
        ),
        (
            {"capture_ids": np.asarray([""] * 10, dtype=np.str_)},
            {},
            "must not be empty",
        ),
    ],
)
def test_rejects_invalid_modeling_partitions(
    training_change: dict[str, object],
    validation_change: dict[str, object],
    message: str,
) -> None:
    training = replace(
        partition_fixture(DatasetPartition.TRAIN, captures_prefix="train"),
        **cast("Any", training_change),
    )
    validation = replace(
        partition_fixture(DatasetPartition.VALIDATION, captures_prefix="validation"),
        **cast("Any", validation_change),
    )
    for partition in (training, validation):
        for array in (
            partition.features,
            partition.categories,
            partition.capture_ids,
            partition.vpn_statuses,
            partition.applications,
        ):
            array.setflags(write=False)

    with pytest.raises(BaselineModelingError, match=message):
        fit_initial_baselines(training, validation)


def test_rejects_writable_partition_arrays() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, captures_prefix="train")
    validation = partition_fixture(DatasetPartition.VALIDATION, captures_prefix="validation")
    writable = training.features.copy()
    training = replace(training, features=writable)

    with pytest.raises(BaselineModelingError, match="must be read-only"):
        fit_initial_baselines(training, validation)


def test_rejects_capture_overlap() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, captures_prefix="shared")
    validation = partition_fixture(DatasetPartition.VALIDATION, captures_prefix="shared")

    with pytest.raises(BaselineModelingError, match="must not overlap"):
        fit_initial_baselines(training, validation)
