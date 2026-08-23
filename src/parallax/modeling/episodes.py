"""Deterministic training-only episode sampling for prototypical models."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from parallax.data import FEATURE_COUNT, DatasetPartition
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.dataset import FeaturePartition

EPISODIC_SAMPLER_SCHEMA_VERSION: Final = "vnat-episodic-sampler-1"
CATEGORY_TO_INDEX: Final = {category: index for index, category in enumerate(CATEGORY_LABELS)}

Int64Vector = npt.NDArray[np.int64]


class EpisodeSamplingError(ValueError):
    """Raised when deterministic training episodes cannot be sampled safely."""


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Paper-aligned episodic-training counts and replay seed."""

    support_examples_per_class: int = 5
    query_examples: int = 512
    episodes: int = 20_000
    random_seed: int = 17

    def __post_init__(self) -> None:
        if self.support_examples_per_class < 1:
            raise EpisodeSamplingError("support examples per class must be positive")
        if self.query_examples < 1:
            raise EpisodeSamplingError("query examples must be positive")
        if self.episodes < 1:
            raise EpisodeSamplingError("episode count must be positive")
        if self.random_seed < 0:
            raise EpisodeSamplingError("random seed must not be negative")

    def as_dict(self) -> dict[str, int]:
        """Return a stable report-ready representation."""
        return {
            "support_examples_per_class": self.support_examples_per_class,
            "query_examples": self.query_examples,
            "episodes": self.episodes,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True, slots=True)
class PrototypeEpisode:
    """One disjoint support/query sample drawn from the training partition."""

    support_features: Tensor
    support_labels: Tensor
    support_indices: Tensor
    query_features: Tensor
    query_labels: Tensor
    query_indices: Tensor


class TrainingEpisodeSampler:
    """Stateful seeded sampler over one verified training partition."""

    def __init__(
        self,
        training: FeaturePartition,
        *,
        config: EpisodeConfig | None = None,
    ) -> None:
        self.config = config or EpisodeConfig()
        _validate_training_partition(training, self.config)

        encoded_labels = np.asarray(
            [CATEGORY_TO_INDEX[label] for label in training.categories],
            dtype=np.int64,
        )
        self._features = torch.tensor(training.features, dtype=torch.float32)
        self._labels = torch.from_numpy(encoded_labels)
        self._indices_by_class = tuple(
            np.flatnonzero(encoded_labels == class_index).astype(np.int64, copy=False)
            for class_index in range(len(CATEGORY_LABELS))
        )
        self._all_indices = np.arange(training.windows, dtype=np.int64)
        self._random = np.random.default_rng(self.config.random_seed)

    def sample(self) -> PrototypeEpisode:
        """Draw the next reproducible episode without support/query overlap."""
        support_indices = np.concatenate(
            tuple(
                cast(
                    "Int64Vector",
                    self._random.choice(
                        class_indices,
                        size=self.config.support_examples_per_class,
                        replace=False,
                    ),
                )
                for class_indices in self._indices_by_class
            )
        )
        query_candidates = np.ones(self._all_indices.shape[0], dtype=np.bool_)
        query_candidates[support_indices] = False
        available_indices = self._all_indices[query_candidates]
        query_indices = cast(
            "Int64Vector",
            self._random.choice(
                available_indices,
                size=self.config.query_examples,
                replace=False,
            ),
        )

        support_tensor = torch.from_numpy(support_indices)
        query_tensor = torch.from_numpy(query_indices)
        return PrototypeEpisode(
            support_features=torch.index_select(self._features, 0, support_tensor),
            support_labels=torch.index_select(self._labels, 0, support_tensor),
            support_indices=support_tensor,
            query_features=torch.index_select(self._features, 0, query_tensor),
            query_labels=torch.index_select(self._labels, 0, query_tensor),
            query_indices=query_tensor,
        )

    def configured_episodes(self) -> Iterator[PrototypeEpisode]:
        """Yield exactly the configured number of successive episodes."""
        for _ in range(self.config.episodes):
            yield self.sample()


def _validate_training_partition(training: FeaturePartition, config: EpisodeConfig) -> None:
    if training.partition is not DatasetPartition.TRAIN:
        raise EpisodeSamplingError("episodic sampling requires the training partition")
    if training.features.ndim != 2 or training.features.shape[0] < 1:
        raise EpisodeSamplingError("feature matrix must be nonempty and two-dimensional")
    if training.features.shape[1] != FEATURE_COUNT:
        raise EpisodeSamplingError(f"feature matrix must contain {FEATURE_COUNT} columns")
    if training.features.dtype != np.float32:
        raise EpisodeSamplingError("feature matrix must use float32 values")
    if not np.isfinite(training.features).all():
        raise EpisodeSamplingError("feature matrix must contain only finite values")
    if training.features.flags.writeable or training.categories.flags.writeable:
        raise EpisodeSamplingError("training arrays must be read-only")
    if training.categories.ndim != 1 or training.categories.shape[0] != training.features.shape[0]:
        raise EpisodeSamplingError("category labels must align with feature rows")
    if set(training.categories.tolist()) != set(CATEGORY_LABELS):
        raise EpisodeSamplingError("training partition must contain exactly the known categories")

    class_counts = {
        category: int(np.count_nonzero(training.categories == category))
        for category in CATEGORY_LABELS
    }
    if any(count < config.support_examples_per_class for count in class_counts.values()):
        raise EpisodeSamplingError("every category must contain enough support examples")
    support_total = config.support_examples_per_class * len(CATEGORY_LABELS)
    if training.windows - support_total < config.query_examples:
        raise EpisodeSamplingError("training partition is too small for disjoint support and query")
