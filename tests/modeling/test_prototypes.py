"""Tests for neural embedding and prototypical-classification primitives."""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from parallax.data import FEATURE_COUNT
from parallax.modeling import (
    DROPOUT_PROBABILITY,
    EMBEDDING_DIMENSION,
    HIDDEN_DIMENSION,
    PROTOTYPE_MODEL_SCHEMA_VERSION,
    TRAFFIC_CATEGORY_COUNT,
    PrototypeEmbeddingNetwork,
    PrototypeModelingError,
    calculate_class_prototypes,
    prototype_probabilities,
    prototypical_cross_entropy,
    squared_euclidean_logits,
)


def test_embedding_network_matches_paper_architecture() -> None:
    torch.manual_seed(17)
    network = PrototypeEmbeddingNetwork()
    network.eval()

    modules = list(network.layers)
    assert PROTOTYPE_MODEL_SCHEMA_VERSION == "vnat-prototype-model-1"
    assert EMBEDDING_DIMENSION == 64
    assert HIDDEN_DIMENSION == 64
    assert DROPOUT_PROBABILITY == 0.25
    assert TRAFFIC_CATEGORY_COUNT == 5
    assert [type(module) for module in modules] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
    ]
    assert isinstance(modules[0], nn.Linear)
    assert (modules[0].in_features, modules[0].out_features) == (FEATURE_COUNT, 64)
    assert isinstance(modules[2], nn.Linear)
    assert (modules[2].in_features, modules[2].out_features) == (64, 64)
    assert isinstance(modules[4], nn.Dropout)
    assert modules[4].p == 0.25
    assert isinstance(modules[5], nn.Linear)
    assert (modules[5].in_features, modules[5].out_features) == (64, 64)
    assert isinstance(modules[7], nn.Dropout)
    assert modules[7].p == 0.25
    assert isinstance(modules[8], nn.Linear)
    assert (modules[8].in_features, modules[8].out_features) == (64, 64)

    embeddings = network(torch.zeros((3, FEATURE_COUNT), dtype=torch.float32))

    assert embeddings.shape == (3, EMBEDDING_DIMENSION)
    assert embeddings.dtype is torch.float32
    assert bool((embeddings >= 0.0).all().item())


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.empty((0, FEATURE_COUNT)), "nonempty and two-dimensional"),
        (torch.empty((FEATURE_COUNT,)), "nonempty and two-dimensional"),
        (torch.empty((1, FEATURE_COUNT - 1)), "must contain 129 columns"),
        (torch.zeros((1, FEATURE_COUNT), dtype=torch.float64), "must use float32"),
        (torch.full((1, FEATURE_COUNT), torch.nan), "only finite"),
    ],
)
def test_embedding_network_rejects_invalid_features(features: Tensor, message: str) -> None:
    with pytest.raises(PrototypeModelingError, match=message):
        PrototypeEmbeddingNetwork()(features)


def test_calculates_differentiable_class_prototypes() -> None:
    embeddings = torch.tensor(
        [[1.0, 3.0], [3.0, 5.0], [10.0, 14.0], [14.0, 18.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    prototypes = calculate_class_prototypes(embeddings, labels, class_count=2)

    assert torch.equal(prototypes, torch.tensor([[2.0, 4.0], [12.0, 16.0]]))
    prototypes.sum().backward()  # type: ignore[no-untyped-call]
    assert embeddings.grad is not None
    assert torch.equal(embeddings.grad, torch.full_like(embeddings, 0.5))


@pytest.mark.parametrize(
    ("embeddings", "labels", "class_count", "message"),
    [
        (
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            2,
            "nonempty two-dimensional",
        ),
        (
            torch.ones((2,), dtype=torch.float32),
            torch.tensor([0, 1]),
            2,
            "nonempty two-dimensional",
        ),
        (
            torch.ones((2, 2), dtype=torch.int64),
            torch.tensor([0, 1]),
            2,
            "must use float32",
        ),
        (
            torch.tensor([[1.0, torch.inf], [2.0, 3.0]]),
            torch.tensor([0, 1]),
            2,
            "only finite",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([[0], [1]]),
            2,
            "one-dimensional",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([0.0, 1.0]),
            2,
            "int64",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([0]),
            2,
            "must align",
        ),
        (
            torch.ones((2, 2)),
            torch.empty((2,), dtype=torch.int64, device="meta"),
            2,
            "share a device",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([0, 1]),
            1,
            "at least two",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([-1, 1]),
            2,
            "valid class indices",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([0, 2]),
            2,
            "valid class indices",
        ),
        (
            torch.ones((2, 2)),
            torch.tensor([0, 0]),
            2,
            "contain every class",
        ),
    ],
)
def test_rejects_invalid_prototype_support(
    embeddings: Tensor,
    labels: Tensor,
    class_count: int,
    message: str,
) -> None:
    with pytest.raises(PrototypeModelingError, match=message):
        calculate_class_prototypes(embeddings, labels, class_count=class_count)


def test_returns_squared_distance_logits_and_probabilities() -> None:
    queries = torch.tensor([[2.0, 4.0], [11.0, 15.0]], requires_grad=True)
    prototypes = torch.tensor([[2.0, 4.0], [12.0, 16.0]], requires_grad=True)

    logits = squared_euclidean_logits(queries, prototypes)
    probabilities = prototype_probabilities(queries, prototypes)

    assert torch.equal(logits, torch.tensor([[0.0, -244.0], [-202.0, -2.0]]))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(probabilities.argmax(dim=1), torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("queries", "prototypes", "message"),
    [
        (torch.ones((2, 3)), torch.ones((2, 2)), "embedding dimension"),
        (
            torch.ones((2, 2), dtype=torch.float32),
            torch.empty((2, 2), dtype=torch.float32, device="meta"),
            "share a device",
        ),
        (
            torch.ones((2, 2), dtype=torch.float32),
            torch.ones((2, 2), dtype=torch.float64),
            "share a dtype",
        ),
        (
            torch.tensor([[torch.finfo(torch.float32).max]], dtype=torch.float32),
            torch.tensor([[-torch.finfo(torch.float32).max]], dtype=torch.float32),
            "must remain finite",
        ),
    ],
)
def test_rejects_invalid_prototype_distance_inputs(
    queries: Tensor,
    prototypes: Tensor,
    message: str,
) -> None:
    with pytest.raises(PrototypeModelingError, match=message):
        squared_euclidean_logits(queries, prototypes)


def test_returns_differentiable_prototypical_loss() -> None:
    queries = torch.tensor([[0.0, 0.0], [2.0, 2.0]], requires_grad=True)
    prototypes = torch.tensor([[0.0, 0.0], [3.0, 3.0]], requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    loss = prototypical_cross_entropy(queries, prototypes, labels)
    expected = functional.cross_entropy(
        torch.tensor([[0.0, -18.0], [-8.0, -2.0]]),
        labels,
    )

    assert float(loss.item()) == pytest.approx(float(expected.item()))
    loss.backward()  # type: ignore[no-untyped-call]
    assert queries.grad is not None and bool(torch.isfinite(queries.grad).all().item())
    assert prototypes.grad is not None and bool(torch.isfinite(prototypes.grad).all().item())


@pytest.mark.parametrize(
    ("make_labels", "message"),
    [
        (lambda: torch.tensor([[0], [1]]), "one-dimensional"),
        (lambda: torch.tensor([0.0, 1.0]), "int64"),
        (lambda: torch.tensor([0]), "must align"),
        (
            lambda: torch.empty((2,), dtype=torch.int64, device="meta"),
            "share a device",
        ),
        (lambda: torch.tensor([-1, 1]), "valid class indices"),
        (lambda: torch.tensor([0, 2]), "valid class indices"),
    ],
)
def test_rejects_invalid_query_labels(
    make_labels: Callable[[], Tensor],
    message: str,
) -> None:
    queries = torch.ones((2, 2), dtype=torch.float32)
    prototypes = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(PrototypeModelingError, match=message):
        prototypical_cross_entropy(queries, prototypes, make_labels())
