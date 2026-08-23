"""Tests for the pre-registered final prototype evaluation procedure."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT, DatasetPartition
from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    OOD_REVIEW_THRESHOLD,
    OOD_STRONG_THRESHOLD,
    PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION,
    FeaturePartition,
    GaussianKDE1D,
    LoadedPrototypeBundle,
    LoadedPrototypeCalibration,
    PrototypeEmbeddingNetwork,
    PrototypeEvaluationError,
    PrototypeInferenceConfig,
    PrototypeInferenceState,
    PrototypeScores,
    PrototypeScoringError,
    RelativeMahalanobisState,
    evaluate_prototype_test,
)


def frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def partition_fixture(partition: DatasetPartition = DatasetPartition.TEST) -> FeaturePartition:
    categories = np.asarray([*CATEGORY_LABELS, *CATEGORY_LABELS], dtype=np.str_)
    return FeaturePartition(
        partition=partition,
        features=frozen(np.zeros((10, FEATURE_COUNT), dtype=np.float32)),
        categories=frozen(categories),
        capture_ids=frozen(
            np.asarray([f"capture-{index // 2}" for index in range(10)], dtype=np.str_)
        ),
        vpn_statuses=frozen(np.asarray(["VPN"] * 10, dtype=np.str_)),
        applications=frozen(np.asarray(["application"] * 10, dtype=np.str_)),
    )


def bundle() -> LoadedPrototypeBundle:
    network = PrototypeEmbeddingNetwork()
    mean = frozen(np.zeros(FEATURE_COUNT, dtype=np.float64))
    scale = frozen(np.ones(FEATURE_COUNT, dtype=np.float64))
    return LoadedPrototypeBundle(
        sha256="a" * 64,
        feature_artifact_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
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


def calibration() -> LoadedPrototypeCalibration:
    identity = torch.eye(EMBEDDING_DIMENSION)
    samples = frozen(np.asarray([0.0, 1.0], dtype=np.float64))
    return LoadedPrototypeCalibration(
        sha256="d" * 64,
        feature_artifact_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        model_bundle_sha256="a" * 64,
        geometry=RelativeMahalanobisState(
            class_means=torch.zeros((5, EMBEDDING_DIMENSION)),
            class_covariances=torch.stack([identity] * 5),
            global_mean=torch.zeros(EMBEDDING_DIMENSION),
            global_covariance=identity,
        ),
        class_kdes=tuple(GaussianKDE1D(samples=samples, bandwidth=0.5) for _ in range(5)),
        calibration_examples_per_class=(2, 2, 2, 2, 2),
        calibration_windows=10,
    )


def score_fixture() -> PrototypeScores:
    probabilities = np.full((10, 5), 0.025, dtype=np.float32)
    expected = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    predicted = expected.copy()
    predicted[-1] = 0
    probabilities[np.arange(10), predicted] = 0.9
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    categories = np.asarray([CATEGORY_LABELS[index] for index in predicted], dtype=np.str_)
    ood_scores = np.asarray(
        [0.01, 0.95, 0.99, 1.0, 0.50, 0.10, 0.94, 0.98, 0.20, 0.30],
        dtype=np.float64,
    )
    for array in (probabilities, predicted, categories, ood_scores):
        array.setflags(write=False)
    distances = frozen(np.arange(10, dtype=np.float64))
    return PrototypeScores(
        category_order=CATEGORY_LABELS,
        class_probabilities=probabilities,
        predicted_class_indices=predicted,
        predicted_categories=categories,
        relative_mahalanobis_distances=distances,
        ood_scores=ood_scores,
    )


def test_evaluates_fixed_closed_set_and_id_false_positive_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "parallax.modeling.evaluation.score_prototype_ood",
        lambda *_: score_fixture(),
    )

    result = evaluate_prototype_test(bundle(), calibration(), partition_fixture())
    payload = result.as_dict()

    assert payload["schema_version"] == PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION
    assert payload["configuration"] == {
        "classification_probabilities": "raw-diagonal-mahalanobis-softmax",
        "probability_temperature_scaling": False,
        "expected_calibration_error_bins": 15,
        "ood_score": "one-minus-fitted-upper-tail-p-value",
        "ood_flag_comparison": "greater-than-or-equal",
        "ood_thresholds": [OOD_REVIEW_THRESHOLD, OOD_STRONG_THRESHOLD],
    }
    assert payload["data"] == {
        "evaluation_partition": "test",
        "test_windows": 10,
        "test_captures": 5,
        "known_traffic_only": True,
        "ood_examples_present": False,
        "model_selection_performed": False,
    }
    assert result.classification_metrics.accuracy == 0.9
    assert result.classification_metrics.categories["FILE_TRANSFER"].recall == 0.5
    assert result.expected_calibration_error == pytest.approx(0.0, abs=1e-6)
    assert result.ood_score_summary.minimum == 0.01
    assert result.ood_score_summary.maximum == 1.0
    assert result.ood_score_summary.mean == pytest.approx(0.597)
    review, strong = result.ood_thresholds
    assert review.flagged_windows == 4
    assert review.false_positive_rate == 0.4
    assert review.categories["VOIP"].flagged_windows == 1
    assert review.categories["VOIP"].false_positive_rate == 0.5
    assert strong.flagged_windows == 2
    assert strong.false_positive_rate == 0.2
    boundary = cast("dict[str, object]", payload["interpretation_boundary"])
    assert boundary["ood_detection_performance_measured"] is False


def test_rejects_non_test_partition() -> None:
    with pytest.raises(PrototypeEvaluationError, match="expected test partition"):
        evaluate_prototype_test(
            bundle(),
            calibration(),
            partition_fixture(DatasetPartition.VALIDATION),
        )


def test_rejects_nonstandard_ece_contract() -> None:
    selected = bundle()
    selected = replace(
        selected,
        inference_config=PrototypeInferenceConfig(
            maximum_support_per_class=2,
            expected_calibration_error_bins=10,
        ),
    )

    with pytest.raises(PrototypeEvaluationError, match="must use 15 ECE bins"):
        evaluate_prototype_test(selected, calibration(), partition_fixture())


def test_wraps_scoring_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> None:
        raise PrototypeScoringError("unavailable")

    monkeypatch.setattr("parallax.modeling.evaluation.score_prototype_ood", fail)

    with pytest.raises(PrototypeEvaluationError, match="evaluation failed: unavailable"):
        evaluate_prototype_test(bundle(), calibration(), partition_fixture())
