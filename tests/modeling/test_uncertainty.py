"""Tests for relative-Mahalanobis geometry and OOD calibration primitives."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    KDE_BANDWIDTH_METHOD,
    GaussianKDE1D,
    OODCalibrationError,
    RelativeMahalanobisState,
    class_conditional_ood_scores,
    fit_class_conditional_kdes,
    fit_gaussian_kde,
    fit_relative_mahalanobis_state,
    relative_mahalanobis_scores,
)


def support_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for class_index in range(len(CATEGORY_LABELS)):
        for sample_index in range(3):
            row = torch.zeros(EMBEDDING_DIMENSION, dtype=torch.float32)
            row[0] = float(class_index * 10 + sample_index)
            row[1] = float(class_index - sample_index)
            rows.append(row)
            labels.append(class_index)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def fitted_state() -> tuple[torch.Tensor, torch.Tensor, RelativeMahalanobisState]:
    embeddings, labels = support_fixture()
    return embeddings, labels, fit_relative_mahalanobis_state(embeddings, labels)


def test_fits_full_covariance_class_and_global_geometry() -> None:
    embeddings, labels, state = fitted_state()

    assert state.class_means.shape == (5, EMBEDDING_DIMENSION)
    assert state.class_covariances.shape == (5, EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
    assert state.global_mean.shape == (EMBEDDING_DIMENSION,)
    assert state.global_covariance.shape == (EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
    assert torch.equal(state.class_means[0, :2], torch.tensor([1.0, -1.0]))
    assert torch.allclose(
        state.global_mean,
        embeddings.mean(dim=0),
    )
    assert torch.allclose(
        state.class_covariances,
        state.class_covariances.transpose(1, 2),
    )
    assert labels.shape == (15,)


def test_computes_relative_distance_as_class_minus_global_distance() -> None:
    embeddings, labels, state = fitted_state()
    queries = embeddings[[1, 7, 14]]
    classes = labels[[1, 7, 14]]

    scores = relative_mahalanobis_scores(queries, classes, state)

    class_precision = torch.linalg.pinv(state.class_covariances, hermitian=True)
    global_precision = torch.linalg.pinv(state.global_covariance, hermitian=True)
    expected: list[torch.Tensor] = []
    for query, class_index in zip(queries, classes, strict=True):
        class_difference = query - state.class_means[class_index]
        global_difference = query - state.global_mean
        expected.append(
            class_difference @ class_precision[class_index] @ class_difference
            - global_difference @ global_precision @ global_difference
        )

    assert torch.allclose(scores, torch.stack(expected))


def test_fits_scott_gaussian_kde_and_monotonic_ood_scores() -> None:
    samples = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    kde = fit_gaussian_kde(samples)
    values = np.asarray([-10.0, 0.0, 10.0], dtype=np.float64)

    scores = kde.cdf(values)

    expected_bandwidth = float(np.std(samples, ddof=1) * samples.size ** (-1.0 / 5.0))
    assert kde.bandwidth == pytest.approx(expected_bandwidth)
    assert scores[0] < scores[1] < scores[2]
    assert scores[1] == pytest.approx(0.5)
    assert kde.as_dict() == {
        "bandwidth_method": KDE_BANDWIDTH_METHOD,
        "sample_count": 5,
        "bandwidth": kde.bandwidth,
    }


def test_fits_and_applies_class_conditional_kdes() -> None:
    calibration_scores = np.concatenate(
        [
            np.asarray([class_index, class_index + 1.0], dtype=np.float64)
            for class_index in range(len(CATEGORY_LABELS))
        ]
    )
    calibration_labels = np.repeat(
        np.arange(len(CATEGORY_LABELS), dtype=np.int64),
        2,
    )
    kdes = fit_class_conditional_kdes(calibration_scores, calibration_labels)
    observed_scores = np.asarray([0.5, 2.5, 4.5], dtype=np.float64)
    predicted = np.asarray([0, 2, 4], dtype=np.int64)

    ood_scores = class_conditional_ood_scores(observed_scores, predicted, kdes)

    assert len(kdes) == len(CATEGORY_LABELS)
    assert np.allclose(ood_scores, np.asarray([0.5, 0.5, 0.5]))
    assert not ood_scores.flags.writeable


@pytest.mark.parametrize(
    ("embeddings", "labels", "message"),
    [
        (torch.empty((0, EMBEDDING_DIMENSION)), torch.empty(0, dtype=torch.int64), "nonempty"),
        (torch.zeros((2, 3)), torch.zeros(2, dtype=torch.int64), "64 dimensions"),
        (
            torch.full((2, EMBEDDING_DIMENSION), torch.nan),
            torch.zeros(2, dtype=torch.int64),
            "finite floating-point",
        ),
        (
            torch.zeros((2, EMBEDDING_DIMENSION)),
            torch.zeros((2, 1), dtype=torch.int64),
            "one-to-one",
        ),
        (
            torch.zeros((2, EMBEDDING_DIMENSION)),
            torch.zeros(2, dtype=torch.int32),
            "int64",
        ),
        (
            torch.zeros((2, EMBEDDING_DIMENSION)),
            torch.tensor([0, 5], dtype=torch.int64),
            "unknown traffic category",
        ),
    ],
)
def test_rejects_invalid_support(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(OODCalibrationError, match=message):
        fit_relative_mahalanobis_state(embeddings, labels)


def test_requires_two_support_examples_per_category() -> None:
    embeddings = torch.zeros((len(CATEGORY_LABELS), EMBEDDING_DIMENSION))
    labels = torch.arange(len(CATEGORY_LABELS), dtype=torch.int64)

    with pytest.raises(OODCalibrationError, match="at least two support examples"):
        fit_relative_mahalanobis_state(embeddings, labels)


def test_rejects_nonfinite_fitted_geometry() -> None:
    embeddings = torch.zeros((10, EMBEDDING_DIMENSION), dtype=torch.float32)
    embeddings[::2, 0] = 1e20
    embeddings[1::2, 0] = -1e20
    labels = torch.repeat_interleave(
        torch.arange(len(CATEGORY_LABELS), dtype=torch.int64),
        2,
    )

    with pytest.raises(OODCalibrationError, match="state must remain finite"):
        fit_relative_mahalanobis_state(embeddings, labels)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("empty", "nonempty"),
        ("dimensions", "64 dimensions"),
        ("nonfinite", "finite floating-point"),
        ("labels_shape", "one-to-one"),
        ("state_shape", "invalid tensor shapes"),
        ("state_nonfinite", "state must contain finite"),
    ],
)
def test_rejects_invalid_relative_distance_inputs(change: str, message: str) -> None:
    embeddings, labels, state = fitted_state()
    queries = embeddings[:2]
    classes = labels[:2]
    if change == "empty":
        queries = torch.empty((0, EMBEDDING_DIMENSION))
        classes = torch.empty(0, dtype=torch.int64)
    elif change == "dimensions":
        queries = torch.zeros((2, 3))
    elif change == "nonfinite":
        queries = torch.full((2, EMBEDDING_DIMENSION), torch.nan)
    elif change == "labels_shape":
        classes = torch.zeros((2, 1), dtype=torch.int64)
    elif change == "state_shape":
        state = replace(state, class_means=state.class_means[:4])
    else:
        state = replace(state, global_mean=torch.full_like(state.global_mean, torch.nan))

    with pytest.raises(OODCalibrationError, match=message):
        relative_mahalanobis_scores(queries, classes, state)


def test_rejects_nonfinite_relative_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings, labels, state = fitted_state()

    def nonfinite_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        assert equation in {"nf,nfg,ng->n", "nf,fg,ng->n"}
        return torch.full((operands[0].shape[0],), torch.nan)

    monkeypatch.setattr("parallax.modeling.uncertainty.torch.einsum", nonfinite_einsum)

    with pytest.raises(OODCalibrationError, match="scores must remain finite"):
        relative_mahalanobis_scores(embeddings[:2], labels[:2], state)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.asarray([], dtype=np.float64), "nonempty"),
        (np.asarray([[1.0, 2.0]], dtype=np.float64), "one-dimensional"),
        (np.asarray([1.0], dtype=np.float64), "at least two"),
        (np.asarray([1.0, 2.0], dtype=np.float32), "float64"),
        (np.asarray([1.0, np.nan], dtype=np.float64), "finite float64"),
        (np.asarray([1.0, 1.0], dtype=np.float64), "positive finite variance"),
    ],
)
def test_rejects_invalid_kde_samples(samples: np.ndarray, message: str) -> None:
    with pytest.raises(OODCalibrationError, match=message):
        fit_gaussian_kde(samples)


@pytest.mark.parametrize(
    ("samples", "bandwidth", "message"),
    [
        (np.asarray([1.0], dtype=np.float64), 1.0, "length two"),
        (np.asarray([1.0, np.nan], dtype=np.float64), 1.0, "finite values"),
        (np.asarray([1.0, 2.0], dtype=np.float64), 0.0, "finite and positive"),
    ],
)
def test_rejects_invalid_kde_state(
    samples: np.ndarray,
    bandwidth: float,
    message: str,
) -> None:
    with pytest.raises(OODCalibrationError, match=message):
        GaussianKDE1D(samples=samples, bandwidth=bandwidth)


def test_rejects_invalid_kde_evaluation_values() -> None:
    kde = fit_gaussian_kde(np.asarray([0.0, 1.0], dtype=np.float64))

    with pytest.raises(OODCalibrationError, match="KDE evaluation values"):
        kde.cdf(np.asarray([0.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("operation", "change", "message"),
    [
        ("fit", "shape", "one-to-one"),
        ("fit", "dtype", "int64"),
        ("fit", "unknown", "unknown traffic category"),
        ("fit", "missing", "at least two samples"),
        ("score", "shape", "one-to-one"),
        ("score", "kdes", "one KDE"),
    ],
)
def test_rejects_invalid_class_conditional_inputs(
    operation: str,
    change: str,
    message: str,
) -> None:
    scores = np.arange(10, dtype=np.float64)
    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    kdes = fit_class_conditional_kdes(scores, labels)
    if change == "shape":
        labels = labels[:-1]
    elif change == "dtype":
        labels = labels.astype(np.int32)
    elif change == "unknown":
        labels[-1] = 5
    elif change == "missing":
        labels[1] = 1
    elif change == "kdes":
        kdes = kdes[:-1]

    with pytest.raises(OODCalibrationError, match=message):
        if operation == "fit":
            fit_class_conditional_kdes(scores, labels)
        else:
            class_conditional_ood_scores(scores, labels, kdes)


def test_rejects_nonfinite_or_out_of_range_ood_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = np.arange(10, dtype=np.float64)
    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    kdes = fit_class_conditional_kdes(scores, labels)

    def invalid_cdf(self: GaussianKDE1D, values: np.ndarray) -> np.ndarray:
        return np.full(values.size, 2.0, dtype=np.float64)

    monkeypatch.setattr(GaussianKDE1D, "cdf", invalid_cdf)

    with pytest.raises(OODCalibrationError, match="within zero and one"):
        class_conditional_ood_scores(scores, labels, kdes)
