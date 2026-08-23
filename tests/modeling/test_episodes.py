"""Tests for deterministic training-only episode sampling."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling import (
    CATEGORY_LABELS,
    CATEGORY_TO_INDEX,
    EPISODIC_SAMPLER_SCHEMA_VERSION,
    EpisodeConfig,
    EpisodeSamplingError,
    FeaturePartition,
    TrainingEpisodeSampler,
)


def training_partition(*, samples_per_category: int = 110) -> FeaturePartition:
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
            features[row, 0] = float(row)
            features[row, category_index + 1] = float(sample_index)
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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"support_examples_per_class": 0}, "support examples per class must be positive"),
        ({"query_examples": 0}, "query examples must be positive"),
        ({"episodes": 0}, "episode count must be positive"),
        ({"random_seed": -1}, "random seed must not be negative"),
    ],
)
def test_rejects_invalid_episode_configuration(
    changes: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(EpisodeSamplingError, match=message):
        EpisodeConfig(**changes)


def test_default_configuration_matches_paper_contract() -> None:
    config = EpisodeConfig()

    assert EPISODIC_SAMPLER_SCHEMA_VERSION == "vnat-episodic-sampler-1"
    assert {category: index for index, category in enumerate(CATEGORY_LABELS)} == CATEGORY_TO_INDEX
    assert config.as_dict() == {
        "support_examples_per_class": 5,
        "query_examples": 512,
        "episodes": 20_000,
        "random_seed": 17,
    }


def test_samples_replayable_disjoint_default_episode() -> None:
    training = training_partition()
    first_sampler = TrainingEpisodeSampler(training)
    replay_sampler = TrainingEpisodeSampler(training)

    first = first_sampler.sample()
    replay = replay_sampler.sample()
    second = first_sampler.sample()

    assert first.support_features.shape == (25, FEATURE_COUNT)
    assert first.support_labels.shape == (25,)
    assert first.query_features.shape == (512, FEATURE_COUNT)
    assert first.query_labels.shape == (512,)
    assert first.support_features.dtype is torch.float32
    assert first.support_labels.dtype is torch.int64
    assert torch.equal(first.support_indices, replay.support_indices)
    assert torch.equal(first.query_indices, replay.query_indices)
    assert torch.equal(first.support_features, replay.support_features)
    assert torch.equal(first.query_features, replay.query_features)
    assert not torch.equal(first.support_indices, second.support_indices)
    assert len(set(first.support_indices.tolist())) == 25
    assert len(set(first.query_indices.tolist())) == 512
    assert set(first.support_indices.tolist()).isdisjoint(first.query_indices.tolist())
    assert torch.equal(
        torch.bincount(first.support_labels, minlength=len(CATEGORY_LABELS)),
        torch.full((len(CATEGORY_LABELS),), 5, dtype=torch.int64),
    )
    assert torch.equal(first.support_features[:, 0], first.support_indices.to(torch.float32))
    assert torch.equal(first.query_features[:, 0], first.query_indices.to(torch.float32))


def test_yields_exact_configured_episode_count() -> None:
    sampler = TrainingEpisodeSampler(
        training_partition(samples_per_category=3),
        config=EpisodeConfig(
            support_examples_per_class=1,
            query_examples=5,
            episodes=2,
            random_seed=29,
        ),
    )

    episodes = tuple(sampler.configured_episodes())

    assert len(episodes) == 2
    assert all(episode.support_features.shape == (5, FEATURE_COUNT) for episode in episodes)
    assert all(episode.query_features.shape == (5, FEATURE_COUNT) for episode in episodes)


@pytest.mark.parametrize(
    ("partition_changes", "config", "message"),
    [
        (
            {"partition": DatasetPartition.VALIDATION},
            EpisodeConfig(query_examples=1),
            "requires the training partition",
        ),
        (
            {"features": np.empty((0, FEATURE_COUNT), dtype=np.float32)},
            EpisodeConfig(query_examples=1),
            "nonempty and two-dimensional",
        ),
        (
            {"features": np.empty((FEATURE_COUNT,), dtype=np.float32)},
            EpisodeConfig(query_examples=1),
            "nonempty and two-dimensional",
        ),
        (
            {"features": np.zeros((15, FEATURE_COUNT - 1), dtype=np.float32)},
            EpisodeConfig(query_examples=1),
            "must contain 129 columns",
        ),
        (
            {"features": np.zeros((15, FEATURE_COUNT), dtype=np.float64)},
            EpisodeConfig(query_examples=1),
            "must use float32",
        ),
        (
            {"features": np.full((15, FEATURE_COUNT), np.nan, dtype=np.float32)},
            EpisodeConfig(query_examples=1),
            "only finite",
        ),
        (
            {"categories": np.asarray(["C2"], dtype=np.str_)},
            EpisodeConfig(query_examples=1),
            "must align",
        ),
        (
            {"categories": np.asarray(["unknown"] * 15, dtype=np.str_)},
            EpisodeConfig(query_examples=1),
            "exactly the known categories",
        ),
        (
            {},
            EpisodeConfig(support_examples_per_class=4, query_examples=1),
            "enough support examples",
        ),
        (
            {},
            EpisodeConfig(support_examples_per_class=3, query_examples=1),
            "too small for disjoint support and query",
        ),
    ],
)
def test_rejects_invalid_training_partition(
    partition_changes: dict[str, object],
    config: EpisodeConfig,
    message: str,
) -> None:
    partition = replace(
        training_partition(samples_per_category=3),
        **cast("Any", partition_changes),
    )
    for array in (partition.features, partition.categories):
        array.setflags(write=False)

    with pytest.raises(EpisodeSamplingError, match=message):
        TrainingEpisodeSampler(partition, config=config)


@pytest.mark.parametrize("writable_field", ["features", "categories"])
def test_rejects_writable_training_arrays(writable_field: str) -> None:
    partition = training_partition(samples_per_category=3)
    writable = getattr(partition, writable_field).copy()
    partition = replace(partition, **{writable_field: writable})

    with pytest.raises(EpisodeSamplingError, match="must be read-only"):
        TrainingEpisodeSampler(partition, config=EpisodeConfig(query_examples=1))
