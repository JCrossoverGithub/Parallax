"""Deterministic training-only optimization for the prototype model."""

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler

from parallax.data import FEATURE_COUNT
from parallax.modeling.dataset import FeaturePartition
from parallax.modeling.episodes import (
    EpisodeConfig,
    TrainingEpisodeSampler,
    _validate_training_partition,
)
from parallax.modeling.prototypes import (
    PrototypeEmbeddingNetwork,
    calculate_class_prototypes,
    prototypical_cross_entropy,
)

PROTOTYPE_TRAINING_SCHEMA_VERSION: Final = "vnat-prototype-training-1"
ADAM_BETA_1: Final = 0.9
ADAM_BETA_2: Final = 0.999
ADAM_EPSILON: Final = 1e-8

Float32Matrix = npt.NDArray[np.float32]
Float64Vector = npt.NDArray[np.float64]


class PrototypeTrainingError(ValueError):
    """Raised when prototype training cannot satisfy its deterministic contract."""


@dataclass(frozen=True, slots=True)
class PrototypeTrainingConfig:
    """Versioned Parallax choices around the paper's episodic procedure."""

    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    torch_seed: int = 17
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise PrototypeTrainingError("learning rate must be finite and positive")
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise PrototypeTrainingError("weight decay must be finite and nonnegative")
        if self.torch_seed < 0:
            raise PrototypeTrainingError("torch seed must not be negative")
        if self.cpu_threads < 1:
            raise PrototypeTrainingError("CPU thread count must be positive")

    def as_dict(self) -> dict[str, object]:
        """Return all documented and paper-derived training choices."""
        return {
            "episode": self.episode.as_dict(),
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta_1": ADAM_BETA_1,
            "adam_beta_2": ADAM_BETA_2,
            "adam_epsilon": ADAM_EPSILON,
            "torch_seed": self.torch_seed,
            "cpu_threads": self.cpu_threads,
            "deterministic_algorithms": True,
            "input_standardization": "standard-scaler-fit-on-training-only",
        }


@dataclass(frozen=True, slots=True)
class PrototypeTrainingResult:
    """Fitted embedding network with training-only preprocessing state."""

    config: PrototypeTrainingConfig
    network: PrototypeEmbeddingNetwork
    feature_mean: Float64Vector
    feature_scale: Float64Vector
    episode_losses: tuple[float, ...]
    training_windows: int

    def standardize(self, features: Float32Matrix) -> Float32Matrix:
        """Apply the immutable training-only scaler to another feature matrix."""
        _validate_transform_features(features)
        standardized = np.asarray(
            (features.astype(np.float64) - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )
        standardized.setflags(write=False)
        return standardized

    def as_dict(self) -> dict[str, object]:
        """Return stable training configuration and loss evidence."""
        return {
            "schema_version": PROTOTYPE_TRAINING_SCHEMA_VERSION,
            "configuration": self.config.as_dict(),
            "data": {
                "fit_partition": "train",
                "training_windows": self.training_windows,
                "feature_count": FEATURE_COUNT,
            },
            "loss": {
                "episodes_completed": len(self.episode_losses),
                "initial": self.episode_losses[0],
                "final": self.episode_losses[-1],
                "minimum": min(self.episode_losses),
            },
        }


def fit_prototype_embedding(
    training: FeaturePartition,
    *,
    config: PrototypeTrainingConfig | None = None,
) -> PrototypeTrainingResult:
    """Fit an embedding network using only seeded episodes from training data."""
    selected_config = config or PrototypeTrainingConfig()
    _validate_training_partition(training, selected_config.episode)
    feature_mean, feature_scale, standardized_features = _fit_standardizer(training.features)
    standardized_training = replace(training, features=standardized_features)
    sampler = TrainingEpisodeSampler(standardized_training, config=selected_config.episode)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    losses: list[float] = []
    try:
        torch.set_num_threads(selected_config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(selected_config.torch_seed)
            network = PrototypeEmbeddingNetwork()
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=selected_config.learning_rate,
                betas=(ADAM_BETA_1, ADAM_BETA_2),
                eps=ADAM_EPSILON,
                weight_decay=selected_config.weight_decay,
            )
            network.train()
            for episode in sampler.configured_episodes():
                optimizer.zero_grad(set_to_none=True)
                support_embeddings = network(episode.support_features)
                query_embeddings = network(episode.query_features)
                prototypes = calculate_class_prototypes(
                    support_embeddings,
                    episode.support_labels,
                )
                loss = prototypical_cross_entropy(
                    query_embeddings,
                    prototypes,
                    episode.query_labels,
                )
                loss_value = float(loss.detach().item())
                if not isfinite(loss_value):
                    raise PrototypeTrainingError("episode loss must remain finite")
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                losses.append(loss_value)
            network.eval()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    return PrototypeTrainingResult(
        config=selected_config,
        network=network,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        episode_losses=tuple(losses),
        training_windows=training.windows,
    )


def _fit_standardizer(
    features: Float32Matrix,
) -> tuple[Float64Vector, Float64Vector, Float32Matrix]:
    scaler = StandardScaler()
    standardized = np.asarray(scaler.fit_transform(features), dtype=np.float32)
    feature_mean = np.asarray(scaler.mean_, dtype=np.float64).copy()
    feature_scale = np.asarray(scaler.scale_, dtype=np.float64).copy()
    for array in (feature_mean, feature_scale, standardized):
        array.setflags(write=False)
    return feature_mean, feature_scale, standardized


def _validate_transform_features(features: Float32Matrix) -> None:
    if features.ndim != 2 or features.shape[0] < 1:
        raise PrototypeTrainingError("feature matrix must be nonempty and two-dimensional")
    if features.shape[1] != FEATURE_COUNT:
        raise PrototypeTrainingError(f"feature matrix must contain {FEATURE_COUNT} columns")
    if features.dtype != np.float32:
        raise PrototypeTrainingError("feature matrix must use float32 values")
    if not np.isfinite(features).all():
        raise PrototypeTrainingError("feature matrix must contain only finite values")
