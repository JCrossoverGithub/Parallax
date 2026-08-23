"""Safe loading of immutable prototype OOD calibration artifacts."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.calibration_report import (
    PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
)
from parallax.modeling.dataset import MODELING_DATASET_SCHEMA_VERSION
from parallax.modeling.prototype_report import PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION
from parallax.modeling.prototypes import EMBEDDING_DIMENSION
from parallax.modeling.uncertainty import (
    KDE_BANDWIDTH_METHOD,
    OOD_CALIBRATION_SCHEMA_VERSION,
    GaussianKDE1D,
    OODCalibrationError,
    RelativeMahalanobisState,
)


class PrototypeCalibrationBundleError(ValueError):
    """Raised when a frozen OOD calibration artifact cannot be trusted."""


@dataclass(frozen=True, slots=True)
class LoadedPrototypeCalibration:
    """Verified relative-Mahalanobis geometry and class-conditional KDEs."""

    sha256: str
    feature_artifact_sha256: str
    split_manifest_sha256: str
    model_bundle_sha256: str
    geometry: RelativeMahalanobisState
    class_kdes: tuple[GaussianKDE1D, ...]
    calibration_examples_per_class: tuple[int, ...]
    calibration_windows: int


def load_prototype_ood_calibration(
    calibration_artifact: str | Path,
    *,
    expected_sha256: str,
    expected_model_bundle_sha256: str,
    expected_feature_artifact_sha256: str,
    expected_split_manifest_sha256: str,
) -> LoadedPrototypeCalibration:
    """Verify a calibration JSON checksum, provenance, and fitted state."""
    path = Path(calibration_artifact)
    if path.suffix.casefold() != ".json":
        raise PrototypeCalibrationBundleError(
            "prototype OOD calibration artifact must use the .json extension"
        )
    encoded = path.read_bytes()
    observed_sha256 = sha256(encoded).hexdigest()
    if observed_sha256 != expected_sha256:
        raise PrototypeCalibrationBundleError(
            "prototype OOD calibration artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        payload = _as_object(json.loads(encoded), name="prototype OOD calibration artifact")
        return _parse_calibration(
            payload,
            observed_sha256=observed_sha256,
            expected_model_bundle_sha256=expected_model_bundle_sha256,
            expected_feature_artifact_sha256=expected_feature_artifact_sha256,
            expected_split_manifest_sha256=expected_split_manifest_sha256,
        )
    except (KeyError, OODCalibrationError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, PrototypeCalibrationBundleError):
            raise
        raise PrototypeCalibrationBundleError(
            f"invalid prototype OOD calibration artifact: {error}"
        ) from error


def _parse_calibration(
    payload: dict[str, object],
    *,
    observed_sha256: str,
    expected_model_bundle_sha256: str,
    expected_feature_artifact_sha256: str,
    expected_split_manifest_sha256: str,
) -> LoadedPrototypeCalibration:
    if payload["schema_version"] != PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION:
        raise PrototypeCalibrationBundleError("unsupported calibration artifact schema version")
    if payload["calibration_schema_version"] != OOD_CALIBRATION_SCHEMA_VERSION:
        raise PrototypeCalibrationBundleError("unsupported OOD calibration schema version")

    provenance = _as_object(payload["provenance"], name="provenance")
    if provenance["modeling_dataset_schema_version"] != MODELING_DATASET_SCHEMA_VERSION:
        raise PrototypeCalibrationBundleError("unsupported modeling dataset schema version")
    feature = _as_object(provenance["feature_artifact"], name="feature artifact provenance")
    manifest = _as_object(provenance["split_manifest"], name="split manifest provenance")
    model = _as_object(provenance["model_bundle"], name="model bundle provenance")
    feature_sha256 = str(feature["sha256"])
    manifest_sha256 = str(manifest["sha256"])
    model_sha256 = str(model["sha256"])
    if feature_sha256 != expected_feature_artifact_sha256:
        raise PrototypeCalibrationBundleError(
            "calibration feature artifact SHA-256 does not match trusted data"
        )
    if manifest_sha256 != expected_split_manifest_sha256:
        raise PrototypeCalibrationBundleError(
            "calibration split manifest SHA-256 does not match trusted data"
        )
    if model_sha256 != expected_model_bundle_sha256:
        raise PrototypeCalibrationBundleError(
            "calibration model bundle SHA-256 does not match the trusted model"
        )
    if model["schema_version"] != PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION:
        raise PrototypeCalibrationBundleError("unsupported prototype model bundle schema version")

    expected_policy = {
        "geometry_fit_partition": "train",
        "density_fit_partition": "calibration",
        "model_selection_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
    }
    if _as_object(payload["evaluation_policy"], name="evaluation policy") != expected_policy:
        raise PrototypeCalibrationBundleError("calibration evaluation policy is not trusted")

    calibration = _as_object(payload["calibration"], name="calibration metadata")
    counts, windows = _parse_calibration_metadata(calibration, model_sha256=model_sha256)
    geometry = _parse_geometry(_as_object(payload["geometry"], name="geometry"))
    class_kdes = _parse_kdes(
        _as_object(payload["class_kdes"], name="class KDEs"),
        expected_counts=counts,
    )
    return LoadedPrototypeCalibration(
        sha256=observed_sha256,
        feature_artifact_sha256=feature_sha256,
        split_manifest_sha256=manifest_sha256,
        model_bundle_sha256=model_sha256,
        geometry=geometry,
        class_kdes=class_kdes,
        calibration_examples_per_class=counts,
        calibration_windows=windows,
    )


def _parse_calibration_metadata(
    payload: dict[str, object],
    *,
    model_sha256: str,
) -> tuple[tuple[int, ...], int]:
    if payload["schema_version"] != OOD_CALIBRATION_SCHEMA_VERSION:
        raise PrototypeCalibrationBundleError("calibration metadata schema version is inconsistent")
    if payload["model_bundle_sha256"] != model_sha256:
        raise PrototypeCalibrationBundleError("calibration metadata references a different model")
    if payload["category_order"] != list(CATEGORY_LABELS):
        raise PrototypeCalibrationBundleError("calibration category order is not trusted")

    configuration = _as_object(payload["configuration"], name="calibration configuration")
    expected_configuration = {
        "support_covariance": "full-population",
        "singular_covariance_handling": "torch-linalg-pseudoinverse",
        "relative_distance": "class-mahalanobis-minus-global-mahalanobis",
        "density_estimator": "class-conditional-univariate-gaussian-kde",
        "bandwidth_method": KDE_BANDWIDTH_METHOD,
        "p_value": "fitted-upper-tail-probability",
        "ood_score": "one-minus-p-value",
    }
    if configuration != expected_configuration:
        raise PrototypeCalibrationBundleError("calibration configuration is not trusted")

    data = _as_object(payload["data"], name="calibration data policy")
    expected_data = {
        "geometry_fit_partition": "train",
        "density_fit_partition": "calibration",
        "calibration_windows": data.get("calibration_windows"),
        "calibration_labels_used": True,
        "model_selection_performed": False,
        "test_evaluated": False,
    }
    if data != expected_data:
        raise PrototypeCalibrationBundleError("calibration data policy is not trusted")
    windows = _exact_int(data["calibration_windows"], name="calibration window count")
    if windows < len(CATEGORY_LABELS) * 2:
        raise PrototypeCalibrationBundleError("calibration window count is too small")

    encoded_counts = _as_object(
        payload["calibration_examples_per_class"],
        name="calibration examples per class",
    )
    if set(encoded_counts) != set(CATEGORY_LABELS):
        raise PrototypeCalibrationBundleError("calibration counts must contain every category")
    counts = tuple(
        _exact_int(encoded_counts[category], name=f"{category} calibration count")
        for category in CATEGORY_LABELS
    )
    if any(count < 2 for count in counts) or sum(counts) != windows:
        raise PrototypeCalibrationBundleError("calibration counts do not match window count")
    return counts, windows


def _parse_geometry(payload: dict[str, object]) -> RelativeMahalanobisState:
    class_count = len(CATEGORY_LABELS)
    class_means = _float32_tensor(
        payload["class_means"],
        name="class means",
        expected_shape=torch.Size((class_count, EMBEDDING_DIMENSION)),
    )
    class_covariances = _float32_tensor(
        payload["class_covariances"],
        name="class covariances",
        expected_shape=torch.Size((class_count, EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)),
    )
    global_mean = _float32_tensor(
        payload["global_mean"],
        name="global mean",
        expected_shape=torch.Size((EMBEDDING_DIMENSION,)),
    )
    global_covariance = _float32_tensor(
        payload["global_covariance"],
        name="global covariance",
        expected_shape=torch.Size((EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)),
    )
    if not torch.allclose(class_covariances, class_covariances.transpose(1, 2)):
        raise PrototypeCalibrationBundleError("class covariances must be symmetric")
    if not torch.allclose(global_covariance, global_covariance.transpose(0, 1)):
        raise PrototypeCalibrationBundleError("global covariance must be symmetric")
    if bool((torch.diagonal(class_covariances, dim1=1, dim2=2) < 0.0).any().item()):
        raise PrototypeCalibrationBundleError("class covariance diagonals must be nonnegative")
    if bool((torch.diagonal(global_covariance) < 0.0).any().item()):
        raise PrototypeCalibrationBundleError("global covariance diagonal must be nonnegative")
    return RelativeMahalanobisState(
        class_means=class_means,
        class_covariances=class_covariances,
        global_mean=global_mean,
        global_covariance=global_covariance,
    )


def _parse_kdes(
    payload: dict[str, object],
    *,
    expected_counts: tuple[int, ...],
) -> tuple[GaussianKDE1D, ...]:
    if set(payload) != set(CATEGORY_LABELS):
        raise PrototypeCalibrationBundleError("class KDEs must contain every traffic category")
    result: list[GaussianKDE1D] = []
    for index, category in enumerate(CATEGORY_LABELS):
        encoded = _as_object(payload[category], name=f"{category} KDE")
        if encoded["bandwidth_method"] != KDE_BANDWIDTH_METHOD:
            raise PrototypeCalibrationBundleError(f"{category} KDE bandwidth method is not trusted")
        sample_count = _exact_int(encoded["sample_count"], name=f"{category} KDE sample count")
        if sample_count != expected_counts[index]:
            raise PrototypeCalibrationBundleError(
                f"{category} KDE sample count does not match calibration metadata"
            )
        samples = _float64_vector(
            encoded["relative_distance_samples"],
            name=f"{category} KDE samples",
            expected_length=sample_count,
        )
        bandwidth = _finite_float(encoded["bandwidth"], name=f"{category} KDE bandwidth")
        expected_bandwidth = float(np.std(samples, ddof=1)) * float(sample_count ** (-1.0 / 5.0))
        if not np.isclose(bandwidth, expected_bandwidth, rtol=1e-12, atol=1e-12):
            raise PrototypeCalibrationBundleError(
                f"{category} KDE bandwidth does not reproduce from its samples"
            )
        result.append(GaussianKDE1D(samples=samples, bandwidth=bandwidth))
    return tuple(result)


def _float32_tensor(value: object, *, name: str, expected_shape: torch.Size) -> Tensor:
    payload = _as_object(value, name=name)
    if payload["dtype"] != "float32":
        raise PrototypeCalibrationBundleError(f"{name} must use float32 values")
    shape = _integer_list(payload["shape"], name=f"{name} shape")
    if tuple(shape) != tuple(expected_shape):
        raise PrototypeCalibrationBundleError(f"{name} does not match its expected shape")
    tensor = torch.tensor(payload["values"], dtype=torch.float32)
    if tensor.shape != expected_shape:
        raise PrototypeCalibrationBundleError(f"{name} values do not match the recorded shape")
    if not bool(torch.isfinite(tensor).all().item()):
        raise PrototypeCalibrationBundleError(f"{name} must contain only finite values")
    return tensor


def _float64_vector(value: object, *, name: str, expected_length: int) -> np.ndarray:
    values = _as_list(value, name=name)
    if len(values) != expected_length or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in values
    ):
        raise PrototypeCalibrationBundleError(
            f"{name} must contain {expected_length} numeric values"
        )
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise PrototypeCalibrationBundleError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PrototypeCalibrationBundleError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise PrototypeCalibrationBundleError(f"{name} must be finite and positive")
    return result


def _integer_list(value: object, *, name: str) -> list[int]:
    values = _as_list(value, name=name)
    if any(type(item) is not int for item in values):
        raise PrototypeCalibrationBundleError(f"{name} must contain only integer values")
    return cast("list[int]", values)


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise PrototypeCalibrationBundleError(f"{name} must be an integer")
    return value


def _as_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PrototypeCalibrationBundleError(f"{name} must be an object")
    return cast("dict[str, object]", value)


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise PrototypeCalibrationBundleError(f"{name} must be an array")
    return value
