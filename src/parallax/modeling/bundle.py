"""Safe, provenance-bound loading for immutable prototype JSON bundles."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from parallax.data import FEATURE_COUNT
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.dataset import MODELING_DATASET_SCHEMA_VERSION
from parallax.modeling.inference import (
    PROTOTYPE_VALIDATION_SCHEMA_VERSION,
    PrototypeInferenceConfig,
    PrototypeInferenceError,
    PrototypeInferenceState,
)
from parallax.modeling.prototype_report import PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION
from parallax.modeling.prototypes import (
    EMBEDDING_DIMENSION,
    PROTOTYPE_MODEL_SCHEMA_VERSION,
    PrototypeEmbeddingNetwork,
)
from parallax.modeling.training import PROTOTYPE_TRAINING_SCHEMA_VERSION

Float32Matrix = npt.NDArray[np.float32]
Float64Vector = npt.NDArray[np.float64]


class PrototypeBundleError(ValueError):
    """Raised when a frozen prototype bundle cannot be trusted or reconstructed."""


@dataclass(frozen=True, slots=True)
class LoadedPrototypeBundle:
    """Verified frozen network, preprocessing, and training-support state."""

    sha256: str
    feature_artifact_sha256: str
    split_manifest_sha256: str
    network: PrototypeEmbeddingNetwork
    feature_mean: Float64Vector
    feature_scale: Float64Vector
    inference_config: PrototypeInferenceConfig
    inference_state: PrototypeInferenceState

    def standardize(self, features: Float32Matrix) -> Float32Matrix:
        """Apply the bundle's immutable training-only standardization state."""
        if features.ndim != 2 or features.shape[0] < 1:
            raise PrototypeBundleError("feature matrix must be nonempty and two-dimensional")
        if features.shape[1] != FEATURE_COUNT:
            raise PrototypeBundleError(f"feature matrix must contain {FEATURE_COUNT} columns")
        if features.dtype != np.float32:
            raise PrototypeBundleError("feature matrix must use float32 values")
        if not np.isfinite(features).all():
            raise PrototypeBundleError("feature matrix must contain only finite values")
        standardized = np.asarray(
            (features.astype(np.float64) - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )
        standardized.setflags(write=False)
        return standardized


def load_prototype_model_bundle(
    model_bundle: str | Path,
    *,
    expected_sha256: str,
    expected_feature_artifact_sha256: str,
    expected_split_manifest_sha256: str,
) -> LoadedPrototypeBundle:
    """Verify a JSON bundle checksum and reconstruct its non-executable state."""
    path = Path(model_bundle)
    if path.suffix.casefold() != ".json":
        raise PrototypeBundleError("prototype model bundle must use the .json extension")
    encoded = path.read_bytes()
    observed_sha256 = sha256(encoded).hexdigest()
    if observed_sha256 != expected_sha256:
        raise PrototypeBundleError(
            "prototype model bundle SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        payload = _as_object(json.loads(encoded), name="prototype model bundle")
        return _parse_bundle(
            payload,
            observed_sha256=observed_sha256,
            expected_feature_artifact_sha256=expected_feature_artifact_sha256,
            expected_split_manifest_sha256=expected_split_manifest_sha256,
        )
    except (KeyError, PrototypeInferenceError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, PrototypeBundleError):
            raise
        raise PrototypeBundleError(f"invalid prototype model bundle: {error}") from error


def _parse_bundle(
    payload: dict[str, object],
    *,
    observed_sha256: str,
    expected_feature_artifact_sha256: str,
    expected_split_manifest_sha256: str,
) -> LoadedPrototypeBundle:
    if payload["schema_version"] != PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION:
        raise PrototypeBundleError("unsupported prototype model bundle schema version")
    provenance = _as_object(payload["provenance"], name="provenance")
    if provenance["modeling_dataset_schema_version"] != MODELING_DATASET_SCHEMA_VERSION:
        raise PrototypeBundleError("unsupported modeling dataset schema version")
    feature_sha256 = str(provenance["feature_artifact_sha256"])
    manifest_sha256 = str(provenance["split_manifest_sha256"])
    if feature_sha256 != expected_feature_artifact_sha256:
        raise PrototypeBundleError("bundle feature artifact SHA-256 does not match trusted data")
    if manifest_sha256 != expected_split_manifest_sha256:
        raise PrototypeBundleError("bundle split manifest SHA-256 does not match trusted data")

    model = _as_object(payload["model"], name="model")
    network = _parse_network(model)
    preprocessing = _as_object(payload["preprocessing"], name="preprocessing")
    feature_mean = _float64_vector(
        preprocessing["feature_mean"],
        name="feature mean",
        expected_length=FEATURE_COUNT,
    )
    feature_scale = _float64_vector(
        preprocessing["feature_scale"],
        name="feature scale",
        expected_length=FEATURE_COUNT,
    )
    if np.any(feature_scale <= 0.0):
        raise PrototypeBundleError("feature scale values must be positive")

    inference = _as_object(payload["inference"], name="inference")
    inference_config = _parse_inference_config(inference)
    inference_state = _parse_inference_state(inference, inference_config)
    return LoadedPrototypeBundle(
        sha256=observed_sha256,
        feature_artifact_sha256=feature_sha256,
        split_manifest_sha256=manifest_sha256,
        network=network,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        inference_config=inference_config,
        inference_state=inference_state,
    )


def _parse_network(payload: dict[str, object]) -> PrototypeEmbeddingNetwork:
    if payload["architecture_schema_version"] != PROTOTYPE_MODEL_SCHEMA_VERSION:
        raise PrototypeBundleError("unsupported prototype architecture schema version")
    if payload["training_schema_version"] != PROTOTYPE_TRAINING_SCHEMA_VERSION:
        raise PrototypeBundleError("unsupported prototype training schema version")
    if payload["category_order"] != list(CATEGORY_LABELS):
        raise PrototypeBundleError("prototype category order does not match the dataset contract")

    network = PrototypeEmbeddingNetwork()
    expected_state = network.state_dict()
    encoded_state = _as_object(payload["state_dict"], name="model state")
    if set(encoded_state) != set(expected_state):
        raise PrototypeBundleError("prototype model state keys do not match the architecture")
    decoded_state = {
        name: _float32_tensor(encoded_state[name], name=name, expected_shape=tensor.shape)
        for name, tensor in expected_state.items()
    }
    network.load_state_dict(decoded_state, strict=True)
    network.requires_grad_(False)
    network.eval()
    return network


def _parse_inference_config(payload: dict[str, object]) -> PrototypeInferenceConfig:
    if payload["validation_schema_version"] != PROTOTYPE_VALIDATION_SCHEMA_VERSION:
        raise PrototypeBundleError("unsupported prototype validation schema version")
    configuration = _as_object(payload["configuration"], name="inference configuration")
    config = PrototypeInferenceConfig(
        maximum_support_per_class=_exact_int(
            configuration["maximum_support_per_class"],
            name="maximum support per class",
        ),
        support_random_seed=_exact_int(
            configuration["support_random_seed"],
            name="support random seed",
        ),
        expected_calibration_error_bins=_exact_int(
            configuration["expected_calibration_error_bins"],
            name="ECE bin count",
        ),
    )
    if configuration != config.as_dict():
        raise PrototypeBundleError("inference configuration does not match its schema contract")
    return config


def _parse_inference_state(
    payload: dict[str, object],
    config: PrototypeInferenceConfig,
) -> PrototypeInferenceState:
    counts = _integer_list(
        payload["support_examples_per_class"],
        name="support examples per class",
    )
    if len(counts) != len(CATEGORY_LABELS):
        raise PrototypeBundleError("support counts must contain every traffic category")
    if any(count < 2 or count > config.maximum_support_per_class for count in counts):
        raise PrototypeBundleError("support counts violate the inference configuration")

    indices = _integer_list(payload["support_indices"], name="support indices")
    if len(indices) != sum(counts):
        raise PrototypeBundleError("support indices do not match the recorded class counts")
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise PrototypeBundleError("support indices must be unique and nonnegative")
    support_indices = torch.tensor(indices, dtype=torch.int64)
    class_means = _float32_tensor(
        payload["class_means"],
        name="class means",
        expected_shape=torch.Size((len(CATEGORY_LABELS), EMBEDDING_DIMENSION)),
    )
    class_variances = _float32_tensor(
        payload["class_variances"],
        name="class variances",
        expected_shape=torch.Size((len(CATEGORY_LABELS), EMBEDDING_DIMENSION)),
    )
    if bool((class_variances < 0.0).any().item()):
        raise PrototypeBundleError("class variances must be nonnegative")
    return PrototypeInferenceState(
        support_indices=support_indices,
        support_examples_per_class=tuple(counts),
        class_means=class_means,
        class_variances=class_variances,
    )


def _float32_tensor(value: object, *, name: str, expected_shape: torch.Size) -> Tensor:
    payload = _as_object(value, name=name)
    if payload["dtype"] != "float32":
        raise PrototypeBundleError(f"{name} must use float32 values")
    shape = _integer_list(payload["shape"], name=f"{name} shape")
    if tuple(shape) != tuple(expected_shape):
        raise PrototypeBundleError(f"{name} does not match its expected shape")
    tensor = torch.tensor(payload["values"], dtype=torch.float32)
    if tensor.shape != expected_shape:
        raise PrototypeBundleError(f"{name} values do not match the recorded shape")
    if not bool(torch.isfinite(tensor).all().item()):
        raise PrototypeBundleError(f"{name} must contain only finite values")
    return tensor


def _float64_vector(value: object, *, name: str, expected_length: int) -> Float64Vector:
    values = _as_list(value, name=name)
    if len(values) != expected_length or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in values
    ):
        raise PrototypeBundleError(f"{name} must contain {expected_length} numeric values")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise PrototypeBundleError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _integer_list(value: object, *, name: str) -> list[int]:
    values = _as_list(value, name=name)
    if any(type(item) is not int for item in values):
        raise PrototypeBundleError(f"{name} must contain only integer values")
    return cast("list[int]", values)


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise PrototypeBundleError(f"{name} must be an integer")
    return value


def _as_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PrototypeBundleError(f"{name} must be an object")
    return cast("dict[str, object]", value)


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise PrototypeBundleError(f"{name} must be an array")
    return value
