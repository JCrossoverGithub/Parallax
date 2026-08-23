"""Tests for deterministic scoring from frozen prototype artifacts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT
from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    GaussianKDE1D,
    LoadedPrototypeBundle,
    LoadedPrototypeCalibration,
    PrototypeEmbeddingNetwork,
    PrototypeInferenceConfig,
    PrototypeInferenceState,
    PrototypeScoringError,
    RelativeMahalanobisState,
    score_prototype_ood,
)

FEATURE_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
MODEL_SHA256 = "c" * 64


def frozen_vector(value: float) -> np.ndarray:
    result = np.full(FEATURE_COUNT, value, dtype=np.float64)
    result.setflags(write=False)
    return result


def model_bundle() -> LoadedPrototypeBundle:
    network = PrototypeEmbeddingNetwork()
    for parameter in network.parameters():
        parameter.data.zero_()
        parameter.requires_grad_(False)
    network.eval()
    class_means = torch.stack(
        [torch.full((EMBEDDING_DIMENSION,), float(index)) for index in range(5)]
    )
    return LoadedPrototypeBundle(
        sha256=MODEL_SHA256,
        feature_artifact_sha256=FEATURE_SHA256,
        split_manifest_sha256=MANIFEST_SHA256,
        network=network,
        feature_mean=frozen_vector(0.0),
        feature_scale=frozen_vector(1.0),
        inference_config=PrototypeInferenceConfig(maximum_support_per_class=2),
        inference_state=PrototypeInferenceState(
            support_indices=torch.arange(10),
            support_examples_per_class=(2, 2, 2, 2, 2),
            class_means=class_means,
            class_variances=torch.ones((5, EMBEDDING_DIMENSION)),
        ),
    )


def calibration_bundle() -> LoadedPrototypeCalibration:
    identity = torch.eye(EMBEDDING_DIMENSION)
    samples = np.asarray([-65.0, -63.0], dtype=np.float64)
    samples.setflags(write=False)
    return LoadedPrototypeCalibration(
        sha256="d" * 64,
        feature_artifact_sha256=FEATURE_SHA256,
        split_manifest_sha256=MANIFEST_SHA256,
        model_bundle_sha256=MODEL_SHA256,
        geometry=RelativeMahalanobisState(
            class_means=torch.zeros((5, EMBEDDING_DIMENSION)),
            class_covariances=torch.stack([identity] * 5),
            global_mean=torch.ones(EMBEDDING_DIMENSION),
            global_covariance=identity,
        ),
        class_kdes=tuple(GaussianKDE1D(samples=samples, bandwidth=1.0) for _ in CATEGORY_LABELS),
        calibration_examples_per_class=(2, 2, 2, 2, 2),
        calibration_windows=10,
    )


def test_returns_immutable_classification_and_ood_scores() -> None:
    features = np.zeros((3, FEATURE_COUNT), dtype=np.float32)

    result = score_prototype_ood(model_bundle(), calibration_bundle(), features)

    assert result.category_order == CATEGORY_LABELS
    assert result.class_probabilities.shape == (3, 5)
    assert np.allclose(result.class_probabilities.sum(axis=1), 1.0)
    assert np.array_equal(result.predicted_class_indices, np.zeros(3, dtype=np.int64))
    assert result.predicted_categories.tolist() == ["STREAMING"] * 3
    assert np.allclose(result.relative_mahalanobis_distances, -64.0)
    assert np.allclose(result.ood_scores, 0.5)
    assert not result.class_probabilities.flags.writeable
    assert not result.predicted_class_indices.flags.writeable
    assert not result.predicted_categories.flags.writeable
    assert not result.relative_mahalanobis_distances.flags.writeable
    assert not result.ood_scores.flags.writeable


def test_scoring_is_deterministic() -> None:
    bundle = model_bundle()
    calibration = calibration_bundle()
    features = np.linspace(
        -1.0,
        1.0,
        num=2 * FEATURE_COUNT,
        dtype=np.float32,
    ).reshape(2, FEATURE_COUNT)

    first = score_prototype_ood(bundle, calibration, features)
    second = score_prototype_ood(bundle, calibration, features)

    assert np.array_equal(first.class_probabilities, second.class_probabilities)
    assert np.array_equal(first.predicted_class_indices, second.predicted_class_indices)
    assert np.array_equal(
        first.relative_mahalanobis_distances, second.relative_mahalanobis_distances
    )
    assert np.array_equal(first.ood_scores, second.ood_scores)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("model", "does not match the model bundle"),
        ("feature", "feature provenance"),
        ("manifest", "split provenance"),
    ],
)
def test_rejects_unbound_artifacts(change: str, message: str) -> None:
    calibration = calibration_bundle()
    if change == "model":
        calibration = replace(calibration, model_bundle_sha256="e" * 64)
    elif change == "feature":
        calibration = replace(calibration, feature_artifact_sha256="e" * 64)
    else:
        calibration = replace(calibration, split_manifest_sha256="e" * 64)

    with pytest.raises(PrototypeScoringError, match=message):
        score_prototype_ood(
            model_bundle(),
            calibration,
            np.zeros((1, FEATURE_COUNT), dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.empty((0, FEATURE_COUNT), dtype=np.float32), "nonempty"),
        (np.zeros((1, 2), dtype=np.float32), "129 columns"),
        (np.zeros((1, FEATURE_COUNT), dtype=np.float64), "float32"),
        (np.full((1, FEATURE_COUNT), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_wraps_invalid_feature_input(features: np.ndarray, message: str) -> None:
    with pytest.raises(PrototypeScoringError, match=message):
        score_prototype_ood(model_bundle(), calibration_bundle(), features)


def test_rejects_invalid_probability_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("parallax.modeling.scoring.CATEGORY_LABELS", CATEGORY_LABELS[:-1])

    with pytest.raises(PrototypeScoringError, match="invalid shape"):
        score_prototype_ood(
            model_bundle(),
            calibration_bundle(),
            np.zeros((1, FEATURE_COUNT), dtype=np.float32),
        )


def test_rejects_nonfinite_probabilities() -> None:
    bundle = model_bundle()
    bundle.inference_state.class_means[0, 0] = float("nan")

    with pytest.raises(PrototypeScoringError, match="probabilities must remain finite"):
        score_prototype_ood(
            bundle,
            calibration_bundle(),
            np.zeros((1, FEATURE_COUNT), dtype=np.float32),
        )
