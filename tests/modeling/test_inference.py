"""Tests for post-training support statistics and validation inference."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    PROTOTYPE_VALIDATION_SCHEMA_VERSION,
    EpisodeConfig,
    FeaturePartition,
    PrototypeInferenceConfig,
    PrototypeInferenceError,
    PrototypeTrainingConfig,
    PrototypeTrainingResult,
    build_prototype_inference_state,
    evaluate_prototype_validation,
    fit_prototype_embedding,
    prototype_class_probabilities,
)


def partition_fixture(
    partition: DatasetPartition,
    *,
    prefix: str,
    samples_per_category: int = 8,
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
        features=features,
        categories=arrays[1],
        capture_ids=arrays[2],
        vpn_statuses=arrays[3],
        applications=arrays[4],
    )


def fitted_training() -> tuple[FeaturePartition, PrototypeTrainingResult]:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    result = fit_prototype_embedding(
        training,
        config=PrototypeTrainingConfig(
            episode=EpisodeConfig(
                support_examples_per_class=1,
                query_examples=10,
                episodes=2,
            )
        ),
    )
    return training, result


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"maximum_support_per_class": 1}, "must be at least two"),
        ({"support_random_seed": -1}, "seed must not be negative"),
        ({"expected_calibration_error_bins": 1}, "bin count must be at least two"),
    ],
)
def test_rejects_invalid_inference_configuration(
    changes: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(PrototypeInferenceError, match=message):
        PrototypeInferenceConfig(**changes)


def test_builds_replayable_training_support_statistics() -> None:
    training, result = fitted_training()
    config = PrototypeInferenceConfig(maximum_support_per_class=3, support_random_seed=29)

    first = build_prototype_inference_state(result, training, config=config)
    replay = build_prototype_inference_state(result, training, config=config)

    assert first.support_examples_per_class == (3, 3, 3, 3, 3)
    assert first.support_indices.shape == (15,)
    assert len(set(first.support_indices.tolist())) == 15
    assert torch.equal(first.support_indices, replay.support_indices)
    assert torch.equal(first.class_means, replay.class_means)
    assert torch.equal(first.class_variances, replay.class_variances)
    assert first.class_means.shape == (5, EMBEDDING_DIMENSION)
    assert first.class_variances.shape == (5, EMBEDDING_DIMENSION)
    assert bool((first.class_variances >= 0.0).all().item())


def test_evaluates_validation_with_diagonal_mahalanobis_probabilities() -> None:
    training, result = fitted_training()
    validation = partition_fixture(DatasetPartition.VALIDATION, prefix="validation")
    config = PrototypeInferenceConfig(maximum_support_per_class=100)

    evaluated = evaluate_prototype_validation(
        result,
        training,
        validation,
        config=config,
    )
    probabilities = prototype_class_probabilities(
        result,
        evaluated.inference_state,
        validation.features,
    )

    assert evaluated.inference_state.support_examples_per_class == (8, 8, 8, 8, 8)
    assert probabilities.shape == (40, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(40))
    assert 0.0 <= evaluated.expected_calibration_error <= 1.0
    assert sum(sum(row) for row in evaluated.validation_metrics.confusion_matrix) == 40

    payload = evaluated.as_dict()
    assert payload["schema_version"] == PROTOTYPE_VALIDATION_SCHEMA_VERSION
    assert payload["configuration"] == {
        "maximum_support_per_class": 100,
        "support_random_seed": 17,
        "classification_distance": "diagonal-mahalanobis",
        "covariance_normalization": "population",
        "singular_covariance_handling": "torch-linalg-pseudoinverse",
        "expected_calibration_error_bins": 15,
    }
    assert payload["data"] == {
        "support_fit_partition": "train",
        "evaluation_partition": "validation",
        "validation_windows": 40,
        "calibration_evaluated": False,
        "test_evaluated": False,
    }
    assert payload["category_order"] == list(CATEGORY_LABELS)
    assert payload["support_examples_per_class"] == {category: 8 for category in CATEGORY_LABELS}
    assert payload["validation_metrics"] == evaluated.validation_metrics.as_dict()
    assert payload["expected_calibration_error"] == evaluated.expected_calibration_error


def test_rejects_training_result_window_mismatch() -> None:
    training, result = fitted_training()
    mismatched = replace(result, training_windows=result.training_windows + 1)

    with pytest.raises(PrototypeInferenceError, match="does not match training window count"):
        build_prototype_inference_state(mismatched, training)


def test_wraps_invalid_training_partition() -> None:
    training, result = fitted_training()
    validation = replace(training, partition=DatasetPartition.VALIDATION)

    with pytest.raises(PrototypeInferenceError, match="invalid training partition"):
        build_prototype_inference_state(result, validation)


def test_wraps_invalid_validation_partition() -> None:
    training, result = fitted_training()

    with pytest.raises(PrototypeInferenceError, match="invalid validation partition"):
        evaluate_prototype_validation(result, training, training)


def test_rejects_capture_overlap() -> None:
    training, result = fitted_training()
    validation = replace(training, partition=DatasetPartition.VALIDATION)

    with pytest.raises(PrototypeInferenceError, match="captures must not overlap"):
        evaluate_prototype_validation(
            result,
            training,
            validation,
        )


def test_rejects_inference_state_without_every_category() -> None:
    training, result = fitted_training()
    state = build_prototype_inference_state(result, training)
    incomplete = replace(
        state,
        class_means=state.class_means[:4],
        class_variances=state.class_variances[:4],
    )

    with pytest.raises(PrototypeInferenceError, match="must contain every traffic category"):
        prototype_class_probabilities(result, incomplete, training.features)


def test_rejects_nonfinite_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    training, result = fitted_training()
    state = build_prototype_inference_state(result, training)

    def nonfinite_softmax(values: torch.Tensor, *, dim: int) -> torch.Tensor:
        assert dim == 1
        return torch.full_like(values, torch.nan)

    monkeypatch.setattr("parallax.modeling.inference.functional.softmax", nonfinite_softmax)

    with pytest.raises(PrototypeInferenceError, match="probabilities must remain finite"):
        prototype_class_probabilities(result, state, training.features)
