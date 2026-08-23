"""Tests for deterministic training-only prototype optimization."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    PROTOTYPE_TRAINING_SCHEMA_VERSION,
    EpisodeConfig,
    EpisodeSamplingError,
    FeaturePartition,
    PrototypeTrainingConfig,
    PrototypeTrainingError,
    fit_prototype_embedding,
)


def training_partition(*, samples_per_category: int = 8) -> FeaturePartition:
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
            features[row, 0] = float(category_index * 10 + sample_index)
            features[row, category_index + 1] = float(sample_index + 1)
            labels.append(category.value)
            captures.append(f"train-{category.value}-{sample_index}.pcap")
            vpn_statuses.append("vpn" if sample_index % 2 else "nonvpn")
            applications.append(f"application-{category_index}")
            row += 1

    category_array = np.asarray(labels, dtype=np.str_)
    capture_array = np.asarray(captures, dtype=np.str_)
    vpn_status_array = np.asarray(vpn_statuses, dtype=np.str_)
    application_array = np.asarray(applications, dtype=np.str_)
    for array in (
        features,
        category_array,
        capture_array,
        vpn_status_array,
        application_array,
    ):
        array.setflags(write=False)

    return FeaturePartition(
        partition=DatasetPartition.TRAIN,
        features=features,
        categories=category_array,
        capture_ids=capture_array,
        vpn_statuses=vpn_status_array,
        applications=application_array,
    )


def small_config() -> PrototypeTrainingConfig:
    return PrototypeTrainingConfig(
        episode=EpisodeConfig(
            support_examples_per_class=1,
            query_examples=10,
            episodes=3,
            random_seed=17,
        ),
        learning_rate=1e-3,
        torch_seed=17,
        cpu_threads=1,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"learning_rate": 0.0}, "learning rate must be finite and positive"),
        ({"learning_rate": float("inf")}, "learning rate must be finite and positive"),
        ({"weight_decay": -1.0}, "weight decay must be finite and nonnegative"),
        ({"weight_decay": float("inf")}, "weight decay must be finite and nonnegative"),
        ({"torch_seed": -1}, "torch seed must not be negative"),
        ({"cpu_threads": 0}, "CPU thread count must be positive"),
    ],
)
def test_rejects_invalid_training_configuration(
    changes: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(PrototypeTrainingError, match=message):
        PrototypeTrainingConfig(**changes)  # type: ignore[arg-type]


def test_default_configuration_records_parallax_training_choices() -> None:
    config = PrototypeTrainingConfig()

    assert PROTOTYPE_TRAINING_SCHEMA_VERSION == "vnat-prototype-training-1"
    assert ADAM_BETA_1 == 0.9
    assert ADAM_BETA_2 == 0.999
    assert ADAM_EPSILON == 1e-8
    assert config.as_dict() == {
        "episode": EpisodeConfig().as_dict(),
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.999,
        "adam_epsilon": 1e-8,
        "torch_seed": 17,
        "cpu_threads": 1,
        "deterministic_algorithms": True,
        "input_standardization": "standard-scaler-fit-on-training-only",
    }


def test_fits_replayable_training_only_embedding() -> None:
    training = training_partition()
    features_before = training.features.copy()
    torch.manual_seed(811)
    random_state_before = torch.random.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()

    first = fit_prototype_embedding(training, config=small_config())
    replay = fit_prototype_embedding(training, config=small_config())

    assert first.episode_losses == replay.episode_losses
    assert len(first.episode_losses) == 3
    assert all(np.isfinite(loss) for loss in first.episode_losses)
    assert not first.network.training
    assert first.training_windows == 40
    assert np.array_equal(training.features, features_before)
    assert torch.equal(torch.random.get_rng_state(), random_state_before)
    assert torch.get_num_threads() == threads_before
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    for name, parameter in first.network.state_dict().items():
        assert torch.equal(parameter, replay.network.state_dict()[name])

    expected_mean = np.mean(training.features, axis=0, dtype=np.float64)
    expected_scale = np.std(training.features, axis=0, dtype=np.float64)
    expected_scale[expected_scale == 0.0] = 1.0
    assert np.allclose(first.feature_mean, expected_mean)
    assert np.allclose(first.feature_scale, expected_scale)
    assert first.feature_mean.dtype == np.float64
    assert first.feature_scale.dtype == np.float64
    assert not first.feature_mean.flags.writeable
    assert not first.feature_scale.flags.writeable

    standardized = first.standardize(training.features)
    assert standardized.dtype == np.float32
    assert not standardized.flags.writeable
    assert np.allclose(np.mean(standardized, axis=0), np.zeros(FEATURE_COUNT), atol=1e-6)

    payload = first.as_dict()
    assert payload["schema_version"] == PROTOTYPE_TRAINING_SCHEMA_VERSION
    assert payload["configuration"] == small_config().as_dict()
    assert payload["data"] == {
        "fit_partition": "train",
        "training_windows": 40,
        "feature_count": 129,
    }
    assert payload["loss"] == {
        "episodes_completed": 3,
        "initial": first.episode_losses[0],
        "final": first.episode_losses[-1],
        "minimum": min(first.episode_losses),
    }


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.empty((0, FEATURE_COUNT), dtype=np.float32), "nonempty and two-dimensional"),
        (np.empty((FEATURE_COUNT,), dtype=np.float32), "nonempty and two-dimensional"),
        (np.empty((1, FEATURE_COUNT - 1), dtype=np.float32), "must contain 129 columns"),
        (np.empty((1, FEATURE_COUNT), dtype=np.float64), "must use float32"),
        (np.full((1, FEATURE_COUNT), np.nan, dtype=np.float32), "only finite"),
    ],
)
def test_rejects_invalid_features_during_standardization(
    features: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    message: str,
) -> None:
    result = fit_prototype_embedding(training_partition(), config=small_config())

    with pytest.raises(PrototypeTrainingError, match=message):
        result.standardize(features)


def test_rejects_nonfinite_episode_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    from parallax.modeling import training as training_module

    def nonfinite_loss(*_args: object) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(training_module, "prototypical_cross_entropy", nonfinite_loss)

    with pytest.raises(PrototypeTrainingError, match="episode loss must remain finite"):
        fit_prototype_embedding(training_partition(), config=small_config())


def test_rejects_nontraining_partition_before_fitting_scaler() -> None:
    validation = replace(training_partition(), partition=DatasetPartition.VALIDATION)

    with pytest.raises(EpisodeSamplingError, match="requires the training partition"):
        fit_prototype_embedding(validation, config=small_config())
