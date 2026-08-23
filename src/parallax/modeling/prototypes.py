"""Neural embedding and prototypical-classification primitives."""

from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from parallax.data import FEATURE_COUNT, TrafficCategory

PROTOTYPE_MODEL_SCHEMA_VERSION: Final = "vnat-prototype-model-1"
EMBEDDING_DIMENSION: Final = 64
HIDDEN_DIMENSION: Final = 64
DROPOUT_PROBABILITY: Final = 0.25
TRAFFIC_CATEGORY_COUNT: Final = len(TrafficCategory)


class PrototypeModelingError(ValueError):
    """Raised when an embedding or prototype operation violates its contract."""


class PrototypeEmbeddingNetwork(nn.Module):
    """Paper-aligned four-layer network that emits a 64-value embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROBABILITY),
            nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROBABILITY),
            nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Embed one nonempty float32 feature matrix."""
        if features.ndim != 2 or features.shape[0] < 1:
            raise PrototypeModelingError("feature tensor must be nonempty and two-dimensional")
        if features.shape[1] != FEATURE_COUNT:
            raise PrototypeModelingError(f"feature tensor must contain {FEATURE_COUNT} columns")
        if features.dtype is not torch.float32:
            raise PrototypeModelingError("feature tensor must use float32 values")
        if not bool(torch.isfinite(features).all().item()):
            raise PrototypeModelingError("feature tensor must contain only finite values")
        return cast("Tensor", self.layers(features))


def calculate_class_prototypes(
    support_embeddings: Tensor,
    support_labels: Tensor,
    *,
    class_count: int = TRAFFIC_CATEGORY_COUNT,
) -> Tensor:
    """Return the mean support embedding for every integer class index."""
    _validate_embedding_matrix(support_embeddings, name="support embeddings")
    if support_labels.ndim != 1:
        raise PrototypeModelingError("support labels must be one-dimensional")
    if support_labels.dtype is not torch.int64:
        raise PrototypeModelingError("support labels must use int64 values")
    if support_labels.shape[0] != support_embeddings.shape[0]:
        raise PrototypeModelingError("support labels must align with support embeddings")
    if support_labels.device != support_embeddings.device:
        raise PrototypeModelingError("support labels and embeddings must share a device")
    if class_count < 2:
        raise PrototypeModelingError("class count must be at least two")
    if int(support_labels.min().item()) < 0 or int(support_labels.max().item()) >= class_count:
        raise PrototypeModelingError("support labels must be valid class indices")

    membership = functional.one_hot(support_labels, num_classes=class_count).to(
        dtype=support_embeddings.dtype
    )
    counts = membership.sum(dim=0)
    if bool((counts == 0).any().item()):
        raise PrototypeModelingError("support set must contain every class")
    return membership.transpose(0, 1) @ support_embeddings / counts.unsqueeze(1)


def squared_euclidean_logits(query_embeddings: Tensor, prototypes: Tensor) -> Tensor:
    """Return negative squared Euclidean distances for prototype classification."""
    _validate_embedding_matrix(query_embeddings, name="query embeddings")
    if query_embeddings.device != prototypes.device:
        raise PrototypeModelingError("queries and prototypes must share a device")
    if query_embeddings.dtype is not prototypes.dtype:
        raise PrototypeModelingError("queries and prototypes must share a dtype")
    _validate_embedding_matrix(prototypes, name="prototypes")
    if query_embeddings.shape[1] != prototypes.shape[1]:
        raise PrototypeModelingError("queries and prototypes must share an embedding dimension")

    differences = query_embeddings[:, None, :] - prototypes[None, :, :]
    logits = -differences.square().sum(dim=2)
    if not bool(torch.isfinite(logits).all().item()):
        raise PrototypeModelingError("squared prototype distances must remain finite")
    return logits


def prototype_probabilities(query_embeddings: Tensor, prototypes: Tensor) -> Tensor:
    """Return normalized class probabilities from prototype distances."""
    return functional.softmax(
        squared_euclidean_logits(query_embeddings, prototypes),
        dim=1,
    )


def prototypical_cross_entropy(
    query_embeddings: Tensor,
    prototypes: Tensor,
    query_labels: Tensor,
) -> Tensor:
    """Return cross-entropy over negative squared prototype distances."""
    logits = squared_euclidean_logits(query_embeddings, prototypes)
    if query_labels.ndim != 1:
        raise PrototypeModelingError("query labels must be one-dimensional")
    if query_labels.dtype is not torch.int64:
        raise PrototypeModelingError("query labels must use int64 values")
    if query_labels.shape[0] != logits.shape[0]:
        raise PrototypeModelingError("query labels must align with query embeddings")
    if query_labels.device != logits.device:
        raise PrototypeModelingError("query labels and embeddings must share a device")
    if int(query_labels.min().item()) < 0 or int(query_labels.max().item()) >= logits.shape[1]:
        raise PrototypeModelingError("query labels must be valid class indices")
    return functional.cross_entropy(logits, query_labels)


def _validate_embedding_matrix(embeddings: Tensor, *, name: str) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise PrototypeModelingError(f"{name} must be a nonempty two-dimensional tensor")
    if embeddings.dtype is not torch.float32:
        raise PrototypeModelingError(f"{name} must use float32 values")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise PrototypeModelingError(f"{name} must contain only finite values")
