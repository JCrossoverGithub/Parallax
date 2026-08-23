"""Tests for calibration-only fitting of prototype OOD distributions."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    CATEGORY_LABELS,
    OOD_CALIBRATION_SCHEMA_VERSION,
    FeaturePartition,
    LoadedPrototypeBundle,
    PrototypeCalibrationError,
    PrototypeCalibrationResult,
    PrototypeEmbeddingNetwork,
    PrototypeInferenceConfig,
    PrototypeInferenceState,
    calibrate_prototype_ood,
)


def partition_fixture(
    partition: DatasetPartition,
    *,
    prefix: str,
    samples_per_category: int = 5,
) -> FeaturePartition:
    row_count = len(CATEGORY_LABELS) * samples_per_category
    features = np.zeros((row_count, FEATURE_COUNT), dtype=np.float32)
    categories: list[str] = []
    captures: list[str] = []
    vpn_statuses: list[str] = []
    applications: list[str] = []
    row = 0
    for class_index, category in enumerate(TrafficCategory):
        for sample_index in range(samples_per_category):
            features[row, 0] = float(class_index * 12 + sample_index)
            features[row, class_index + 1] = float((sample_index + 1) ** 2)
            features[row, 8] = float((class_index + 1) * (sample_index + 2))
            categories.append(category.value)
            captures.append(f"{prefix}-{category.value}-{sample_index}.pcap")
            vpn_statuses.append("vpn" if sample_index % 2 else "nonvpn")
            applications.append(f"application-{class_index}")
            row += 1
    arrays = (
        features,
        np.asarray(categories, dtype=np.str_),
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


def bundle_fixture(training: FeaturePartition) -> LoadedPrototypeBundle:
    torch.manual_seed(17)
    network = PrototypeEmbeddingNetwork()
    network.eval()
    network.requires_grad_(False)
    selected_by_class = tuple(
        np.flatnonzero(training.categories == category)[:3] for category in CATEGORY_LABELS
    )
    indices = np.concatenate(selected_by_class).astype(np.int64)
    labels = torch.repeat_interleave(torch.arange(5, dtype=torch.int64), 3)
    with torch.no_grad():
        embeddings = network(torch.from_numpy(training.features[indices].copy()))
        membership = functional.one_hot(labels, num_classes=5).to(dtype=torch.float32)
        counts = membership.sum(dim=0)
        means = membership.transpose(0, 1) @ embeddings / counts.unsqueeze(1)
        residuals = embeddings - means[labels]
        variances = membership.transpose(0, 1) @ residuals.square()
        variances = variances / counts.unsqueeze(1)
    mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
    scale = np.ones(FEATURE_COUNT, dtype=np.float64)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return LoadedPrototypeBundle(
        sha256="c" * 64,
        feature_artifact_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        network=network,
        feature_mean=mean,
        feature_scale=scale,
        inference_config=PrototypeInferenceConfig(maximum_support_per_class=3),
        inference_state=PrototypeInferenceState(
            support_indices=torch.from_numpy(indices),
            support_examples_per_class=(3, 3, 3, 3, 3),
            class_means=means,
            class_variances=variances,
        ),
    )


def fitted_calibration() -> tuple[
    FeaturePartition,
    FeaturePartition,
    LoadedPrototypeBundle,
    PrototypeCalibrationResult,
]:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(
        DatasetPartition.CALIBRATION,
        prefix="calibration",
    )
    bundle = bundle_fixture(training)
    return (
        training,
        calibration,
        bundle,
        calibrate_prototype_ood(
            bundle,
            training,
            calibration,
        ),
    )


def test_fits_replayable_calibration_only_ood_state() -> None:
    training, calibration, bundle, first = fitted_calibration()
    replay = calibrate_prototype_ood(bundle, training, calibration)

    assert first.bundle is bundle
    assert first.calibration_windows == 25
    assert first.calibration_examples_per_class == (5, 5, 5, 5, 5)
    assert first.geometry.class_covariances.shape == (5, 64, 64)
    assert first.geometry.global_covariance.shape == (64, 64)
    assert first.relative_scores.shape == (25,)
    assert not first.relative_scores.flags.writeable
    assert len(first.class_kdes) == 5
    assert all(kde.samples.size == 5 for kde in first.class_kdes)
    assert torch.equal(first.geometry.class_means, replay.geometry.class_means)
    assert torch.equal(first.geometry.class_covariances, replay.geometry.class_covariances)
    assert np.array_equal(first.relative_scores, replay.relative_scores)
    assert [kde.bandwidth for kde in first.class_kdes] == [
        kde.bandwidth for kde in replay.class_kdes
    ]


def test_records_calibration_without_model_selection_or_test_access() -> None:
    _, _, _, result = fitted_calibration()

    payload = result.as_dict()

    assert payload["schema_version"] == OOD_CALIBRATION_SCHEMA_VERSION
    assert payload["model_bundle_sha256"] == "c" * 64
    assert payload["configuration"] == {
        "support_covariance": "full-population",
        "singular_covariance_handling": "torch-linalg-pseudoinverse",
        "relative_distance": "class-mahalanobis-minus-global-mahalanobis",
        "density_estimator": "class-conditional-univariate-gaussian-kde",
        "bandwidth_method": "scott",
        "p_value": "fitted-upper-tail-probability",
        "ood_score": "one-minus-p-value",
    }
    assert payload["data"] == {
        "geometry_fit_partition": "train",
        "density_fit_partition": "calibration",
        "calibration_windows": 25,
        "calibration_labels_used": True,
        "model_selection_performed": False,
        "test_evaluated": False,
    }
    assert payload["category_order"] == list(CATEGORY_LABELS)
    assert payload["calibration_examples_per_class"] == {
        category: 5 for category in CATEGORY_LABELS
    }
    summaries = payload["relative_score_summary"]
    assert isinstance(summaries, dict)
    assert set(summaries) == set(CATEGORY_LABELS)
    for summary in summaries.values():
        assert isinstance(summary, dict)
        assert summary["minimum"] <= summary["mean"] <= summary["maximum"]
        assert summary["kde"]["sample_count"] == 5


@pytest.mark.parametrize(
    ("partition", "message"),
    [
        (DatasetPartition.CALIBRATION, "expected train partition"),
        (DatasetPartition.VALIDATION, "expected calibration partition"),
    ],
)
def test_rejects_wrong_partition_roles(
    partition: DatasetPartition,
    message: str,
) -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(
        DatasetPartition.CALIBRATION,
        prefix="calibration",
    )
    bundle = bundle_fixture(training)
    if partition is DatasetPartition.CALIBRATION:
        training = replace(training, partition=partition)
    else:
        calibration = replace(calibration, partition=partition)

    with pytest.raises(PrototypeCalibrationError, match=message):
        calibrate_prototype_ood(bundle, training, calibration)


def test_rejects_training_calibration_capture_overlap() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="same")
    calibration = partition_fixture(DatasetPartition.CALIBRATION, prefix="same")

    with pytest.raises(PrototypeCalibrationError, match="captures must not overlap"):
        calibrate_prototype_ood(bundle_fixture(training), training, calibration)


def test_rejects_support_indices_outside_training_partition() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(DatasetPartition.CALIBRATION, prefix="calibration")
    bundle = bundle_fixture(training)
    state = replace(
        bundle.inference_state,
        support_indices=torch.full_like(
            bundle.inference_state.support_indices,
            training.windows,
        ),
    )

    with pytest.raises(PrototypeCalibrationError, match="indices exceed"):
        calibrate_prototype_ood(
            replace(bundle, inference_state=state),
            training,
            calibration,
        )


def test_rejects_support_indices_with_wrong_training_categories() -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(DatasetPartition.CALIBRATION, prefix="calibration")
    bundle = bundle_fixture(training)
    state = replace(
        bundle.inference_state,
        support_indices=torch.arange(15, dtype=torch.int64),
    )

    with pytest.raises(PrototypeCalibrationError, match="do not match training categories"):
        calibrate_prototype_ood(
            replace(bundle, inference_state=state),
            training,
            calibration,
        )


@pytest.mark.parametrize("statistic", ["means", "variances"])
def test_rejects_support_statistics_that_do_not_reproduce(statistic: str) -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(DatasetPartition.CALIBRATION, prefix="calibration")
    bundle = bundle_fixture(training)
    state = bundle.inference_state
    if statistic == "means":
        state = replace(state, class_means=state.class_means + 1.0)
    else:
        state = replace(state, class_variances=state.class_variances + 1.0)

    with pytest.raises(PrototypeCalibrationError, match=f"class {statistic} do not reproduce"):
        calibrate_prototype_ood(
            replace(bundle, inference_state=state),
            training,
            calibration,
        )


def test_wraps_nonvariable_calibration_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    training = partition_fixture(DatasetPartition.TRAIN, prefix="train")
    calibration = partition_fixture(DatasetPartition.CALIBRATION, prefix="calibration")
    bundle = bundle_fixture(training)

    def constant_scores(
        embeddings: torch.Tensor,
        class_indices: torch.Tensor,
        state: object,
    ) -> torch.Tensor:
        assert embeddings.shape[0] == class_indices.shape[0]
        assert state is not None
        return torch.zeros(embeddings.shape[0])

    monkeypatch.setattr(
        "parallax.modeling.calibration.relative_mahalanobis_scores",
        constant_scores,
    )

    with pytest.raises(PrototypeCalibrationError, match="positive finite variance"):
        calibrate_prototype_ood(bundle, training, calibration)
