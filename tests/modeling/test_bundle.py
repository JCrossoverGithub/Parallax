"""Tests for safe loading of immutable prototype JSON bundles."""

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from parallax.data import FEATURE_COUNT
from parallax.modeling import (
    CATEGORY_LABELS,
    MODELING_DATASET_SCHEMA_VERSION,
    PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    PROTOTYPE_MODEL_SCHEMA_VERSION,
    PROTOTYPE_TRAINING_SCHEMA_VERSION,
    PROTOTYPE_VALIDATION_SCHEMA_VERSION,
    PrototypeBundleError,
    PrototypeEmbeddingNetwork,
    PrototypeInferenceConfig,
    load_prototype_model_bundle,
)

FEATURE_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64


def tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def valid_payload() -> dict[str, object]:
    torch.manual_seed(17)
    network = PrototypeEmbeddingNetwork()
    config = PrototypeInferenceConfig(maximum_support_per_class=2)
    return {
        "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
        "provenance": {
            "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
            "feature_artifact_sha256": FEATURE_SHA256,
            "split_manifest_sha256": MANIFEST_SHA256,
        },
        "model": {
            "architecture_schema_version": PROTOTYPE_MODEL_SCHEMA_VERSION,
            "training_schema_version": PROTOTYPE_TRAINING_SCHEMA_VERSION,
            "category_order": list(CATEGORY_LABELS),
            "state_dict": {
                name: tensor_payload(tensor) for name, tensor in network.state_dict().items()
            },
        },
        "preprocessing": {
            "feature_mean": [1.0] * FEATURE_COUNT,
            "feature_scale": [2.0] * FEATURE_COUNT,
        },
        "inference": {
            "validation_schema_version": PROTOTYPE_VALIDATION_SCHEMA_VERSION,
            "configuration": config.as_dict(),
            "support_indices": list(range(10)),
            "support_examples_per_class": [2] * len(CATEGORY_LABELS),
            "class_means": tensor_payload(torch.zeros((5, 64))),
            "class_variances": tensor_payload(torch.ones((5, 64))),
        },
    }


def write_bundle(path: Path, payload: object) -> str:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def load(path: Path, payload: object | None = None):  # type: ignore[no-untyped-def]
    bundle_sha256 = write_bundle(path, valid_payload() if payload is None else payload)
    return load_prototype_model_bundle(
        path,
        expected_sha256=bundle_sha256,
        expected_feature_artifact_sha256=FEATURE_SHA256,
        expected_split_manifest_sha256=MANIFEST_SHA256,
    )


def nested(payload: dict[str, object], key: str) -> dict[str, object]:
    return cast("dict[str, object]", payload[key])


def test_loads_frozen_network_preprocessing_and_inference_state(tmp_path: Path) -> None:
    path = tmp_path / "prototype-model.json"
    loaded = load(path)
    features = np.full((2, FEATURE_COUNT), 3.0, dtype=np.float32)

    standardized = loaded.standardize(features)

    assert loaded.sha256 == sha256(path.read_bytes()).hexdigest()
    assert loaded.feature_artifact_sha256 == FEATURE_SHA256
    assert loaded.split_manifest_sha256 == MANIFEST_SHA256
    assert not loaded.network.training
    assert all(not parameter.requires_grad for parameter in loaded.network.parameters())
    assert np.array_equal(standardized, np.ones_like(features))
    assert not standardized.flags.writeable
    assert not loaded.feature_mean.flags.writeable
    assert not loaded.feature_scale.flags.writeable
    assert loaded.inference_config.maximum_support_per_class == 2
    assert loaded.inference_state.support_indices.tolist() == list(range(10))
    assert loaded.inference_state.support_examples_per_class == (2, 2, 2, 2, 2)
    assert loaded.inference_state.class_means.shape == (5, 64)
    assert loaded.inference_state.class_variances.shape == (5, 64)


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(PrototypeBundleError, match=r"must use the \.json extension"):
        load_prototype_model_bundle(
            tmp_path / "model.txt",
            expected_sha256="a" * 64,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    write_bundle(path, valid_payload())

    with pytest.raises(PrototypeBundleError, match="bundle SHA-256 mismatch"):
        load_prototype_model_bundle(
            path,
            expected_sha256="0" * 64,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("root", "bundle must be an object"),
        ("schema", "unsupported prototype model bundle"),
        ("provenance_type", "provenance must be an object"),
        ("dataset_schema", "unsupported modeling dataset"),
        ("feature_hash", "feature artifact SHA-256"),
        ("manifest_hash", "split manifest SHA-256"),
        ("missing", "invalid prototype model bundle"),
    ],
)
def test_rejects_invalid_bundle_envelope(tmp_path: Path, change: str, message: str) -> None:
    payload: object = valid_payload()
    if change == "root":
        payload = []
    else:
        root = cast("dict[str, object]", payload)
        provenance = nested(root, "provenance")
        if change == "schema":
            root["schema_version"] = "future"
        elif change == "provenance_type":
            root["provenance"] = []
        elif change == "dataset_schema":
            provenance["modeling_dataset_schema_version"] = "future"
        elif change == "feature_hash":
            provenance["feature_artifact_sha256"] = "c" * 64
        elif change == "manifest_hash":
            provenance["split_manifest_sha256"] = "c" * 64
        else:
            del root["model"]

    with pytest.raises(PrototypeBundleError, match=message):
        load(tmp_path / "model.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("architecture", "unsupported prototype architecture"),
        ("training", "unsupported prototype training"),
        ("categories", "category order"),
        ("state_type", "model state must be an object"),
        ("state_keys", "state keys"),
        ("dtype", "must use float32"),
        ("shape_type", "shape must be an array"),
        ("shape_value", "shape must contain only integer"),
        ("shape", "expected shape"),
        ("values_shape", "values do not match"),
        ("nonfinite", "finite values"),
    ],
)
def test_rejects_invalid_network_state(tmp_path: Path, change: str, message: str) -> None:
    payload = valid_payload()
    model = nested(payload, "model")
    state = nested(model, "state_dict")
    first_name = next(iter(state))
    first_tensor = nested(state, first_name)
    if change == "architecture":
        model["architecture_schema_version"] = "future"
    elif change == "training":
        model["training_schema_version"] = "future"
    elif change == "categories":
        model["category_order"] = []
    elif change == "state_type":
        model["state_dict"] = []
    elif change == "state_keys":
        del state[first_name]
    elif change == "dtype":
        first_tensor["dtype"] = "float64"
    elif change == "shape_type":
        first_tensor["shape"] = 1
    elif change == "shape_value":
        first_tensor["shape"] = [True]
    elif change == "shape":
        first_tensor["shape"] = [1]
    elif change == "values_shape":
        first_tensor["values"] = []
    else:
        values = cast("list[list[float]]", first_tensor["values"])
        values[0][0] = float("nan")

    with pytest.raises(PrototypeBundleError, match=message):
        load(tmp_path / "model.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("preprocessing_type", "preprocessing must be an object"),
        ("mean_type", "feature mean must be an array"),
        ("mean_length", "129 numeric values"),
        ("mean_value", "129 numeric values"),
        ("mean_nonfinite", "finite values"),
        ("scale_zero", "scale values must be positive"),
    ],
)
def test_rejects_invalid_preprocessing(tmp_path: Path, change: str, message: str) -> None:
    payload = valid_payload()
    preprocessing = nested(payload, "preprocessing")
    if change == "preprocessing_type":
        payload["preprocessing"] = []
    elif change == "mean_type":
        preprocessing["feature_mean"] = 1
    elif change == "mean_length":
        preprocessing["feature_mean"] = []
    elif change == "mean_value":
        preprocessing["feature_mean"] = [True] * FEATURE_COUNT
    elif change == "mean_nonfinite":
        preprocessing["feature_mean"] = [float("inf")] * FEATURE_COUNT
    else:
        preprocessing["feature_scale"] = [0.0] * FEATURE_COUNT

    with pytest.raises(PrototypeBundleError, match=message):
        load(tmp_path / "model.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("inference_type", "inference must be an object"),
        ("schema", "unsupported prototype validation"),
        ("configuration_type", "configuration must be an object"),
        ("config_integer", "must be an integer"),
        ("config_value", "must be at least two"),
        ("config_contract", "configuration does not match"),
        ("counts_type", "must be an array"),
        ("counts_integer", "only integer"),
        ("counts_length", "every traffic category"),
        ("counts_value", "violate the inference"),
        ("indices_length", "do not match"),
        ("indices_value", "unique and nonnegative"),
        ("indices_duplicate", "unique and nonnegative"),
        ("variance_negative", "must be nonnegative"),
    ],
)
def test_rejects_invalid_inference_state(tmp_path: Path, change: str, message: str) -> None:
    payload = valid_payload()
    inference = nested(payload, "inference")
    configuration = nested(inference, "configuration")
    if change == "inference_type":
        payload["inference"] = []
    elif change == "schema":
        inference["validation_schema_version"] = "future"
    elif change == "configuration_type":
        inference["configuration"] = []
    elif change == "config_integer":
        configuration["maximum_support_per_class"] = True
    elif change == "config_value":
        configuration["maximum_support_per_class"] = 1
    elif change == "config_contract":
        configuration["classification_distance"] = "euclidean"
    elif change == "counts_type":
        inference["support_examples_per_class"] = 2
    elif change == "counts_integer":
        inference["support_examples_per_class"] = [True] * 5
    elif change == "counts_length":
        inference["support_examples_per_class"] = [2] * 4
    elif change == "counts_value":
        inference["support_examples_per_class"] = [3] * 5
    elif change == "indices_length":
        inference["support_indices"] = [0]
    elif change == "indices_value":
        inference["support_indices"] = [*list(range(9)), -1]
    elif change == "indices_duplicate":
        inference["support_indices"] = [0] * 10
    else:
        variances = nested(inference, "class_variances")
        values = cast("list[list[float]]", variances["values"])
        values[0][0] = -1.0

    with pytest.raises(PrototypeBundleError, match=message):
        load(tmp_path / "model.json", payload)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.empty((0, FEATURE_COUNT), dtype=np.float32), "nonempty"),
        (np.zeros((1, 3), dtype=np.float32), "129 columns"),
        (np.zeros((1, FEATURE_COUNT), dtype=np.float64), "float32"),
        (np.full((1, FEATURE_COUNT), np.nan, dtype=np.float32), "finite values"),
    ],
)
def test_rejects_invalid_standardization_input(
    tmp_path: Path,
    features: np.ndarray,
    message: str,
) -> None:
    loaded = load(tmp_path / "model.json")

    with pytest.raises(PrototypeBundleError, match=message):
        loaded.standardize(features)


def test_wraps_json_decode_failure(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("{", encoding="utf-8")
    digest = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(PrototypeBundleError, match="invalid prototype model bundle"):
        load_prototype_model_bundle(
            path,
            expected_sha256=digest,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )
