"""Tests for safe loading of frozen prototype OOD calibration artifacts."""

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from parallax.modeling import (
    CATEGORY_LABELS,
    EMBEDDING_DIMENSION,
    KDE_BANDWIDTH_METHOD,
    MODELING_DATASET_SCHEMA_VERSION,
    OOD_CALIBRATION_SCHEMA_VERSION,
    PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
    PrototypeCalibrationBundleError,
    load_prototype_ood_calibration,
)

FEATURE_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
MODEL_SHA256 = "c" * 64


def tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def valid_payload() -> dict[str, object]:
    counts = {category: 2 for category in CATEGORY_LABELS}
    class_kdes = {}
    for index, category in enumerate(CATEGORY_LABELS):
        samples = np.asarray([float(index), float(index + 1)], dtype=np.float64)
        bandwidth = float(np.std(samples, ddof=1)) * float(samples.size ** (-1.0 / 5.0))
        class_kdes[category] = {
            "bandwidth_method": KDE_BANDWIDTH_METHOD,
            "sample_count": 2,
            "bandwidth": bandwidth,
            "relative_distance_samples": samples.tolist(),
        }
    identity = torch.eye(EMBEDDING_DIMENSION)
    return {
        "schema_version": PROTOTYPE_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            "modeling_dataset_schema_version": MODELING_DATASET_SCHEMA_VERSION,
            "feature_artifact": {
                "file_name": "features.parquet",
                "file_size_bytes": 1,
                "sha256": FEATURE_SHA256,
            },
            "split_manifest": {
                "file_name": "capture-splits.json",
                "file_size_bytes": 1,
                "sha256": MANIFEST_SHA256,
            },
            "model_bundle": {
                "file_name": "prototype-model.json",
                "file_size_bytes": 1,
                "sha256": MODEL_SHA256,
                "schema_version": PROTOTYPE_MODEL_BUNDLE_SCHEMA_VERSION,
            },
        },
        "evaluation_policy": {
            "geometry_fit_partition": "train",
            "density_fit_partition": "calibration",
            "model_selection_performed": False,
            "validation_evaluated": False,
            "test_evaluated": False,
        },
        "calibration_schema_version": OOD_CALIBRATION_SCHEMA_VERSION,
        "calibration": {
            "schema_version": OOD_CALIBRATION_SCHEMA_VERSION,
            "model_bundle_sha256": MODEL_SHA256,
            "configuration": {
                "support_covariance": "full-population",
                "singular_covariance_handling": "torch-linalg-pseudoinverse",
                "relative_distance": "class-mahalanobis-minus-global-mahalanobis",
                "density_estimator": "class-conditional-univariate-gaussian-kde",
                "bandwidth_method": KDE_BANDWIDTH_METHOD,
                "p_value": "fitted-upper-tail-probability",
                "ood_score": "one-minus-p-value",
            },
            "data": {
                "geometry_fit_partition": "train",
                "density_fit_partition": "calibration",
                "calibration_windows": 10,
                "calibration_labels_used": True,
                "model_selection_performed": False,
                "test_evaluated": False,
            },
            "category_order": list(CATEGORY_LABELS),
            "calibration_examples_per_class": counts,
            "relative_score_summary": {},
        },
        "geometry": {
            "class_means": tensor_payload(torch.zeros((5, EMBEDDING_DIMENSION))),
            "class_covariances": tensor_payload(torch.stack([identity] * 5)),
            "global_mean": tensor_payload(torch.zeros(EMBEDDING_DIMENSION)),
            "global_covariance": tensor_payload(identity),
        },
        "class_kdes": class_kdes,
    }


def nested(payload: dict[str, object], key: str) -> dict[str, object]:
    return cast("dict[str, object]", payload[key])


def write_artifact(path: Path, payload: object) -> str:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def load(path: Path, payload: object | None = None):  # type: ignore[no-untyped-def]
    digest = write_artifact(path, valid_payload() if payload is None else payload)
    return load_prototype_ood_calibration(
        path,
        expected_sha256=digest,
        expected_model_bundle_sha256=MODEL_SHA256,
        expected_feature_artifact_sha256=FEATURE_SHA256,
        expected_split_manifest_sha256=MANIFEST_SHA256,
    )


def test_loads_verified_geometry_and_kdes(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"

    loaded = load(path)

    assert loaded.sha256 == sha256(path.read_bytes()).hexdigest()
    assert loaded.feature_artifact_sha256 == FEATURE_SHA256
    assert loaded.split_manifest_sha256 == MANIFEST_SHA256
    assert loaded.model_bundle_sha256 == MODEL_SHA256
    assert loaded.calibration_examples_per_class == (2, 2, 2, 2, 2)
    assert loaded.calibration_windows == 10
    assert loaded.geometry.class_means.shape == (5, EMBEDDING_DIMENSION)
    assert loaded.geometry.class_covariances.shape == (5, 64, 64)
    assert loaded.geometry.global_mean.shape == (64,)
    assert loaded.geometry.global_covariance.shape == (64, 64)
    assert len(loaded.class_kdes) == len(CATEGORY_LABELS)
    assert all(not kde.samples.flags.writeable for kde in loaded.class_kdes)


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(PrototypeCalibrationBundleError, match=r"must use the \.json extension"):
        load_prototype_ood_calibration(
            tmp_path / "calibration.txt",
            expected_sha256="d" * 64,
            expected_model_bundle_sha256=MODEL_SHA256,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    write_artifact(path, valid_payload())

    with pytest.raises(PrototypeCalibrationBundleError, match="artifact SHA-256 mismatch"):
        load_prototype_ood_calibration(
            path,
            expected_sha256="0" * 64,
            expected_model_bundle_sha256=MODEL_SHA256,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("root", "artifact must be an object"),
        ("schema", "unsupported calibration artifact"),
        ("calibration_schema", "unsupported OOD calibration"),
        ("provenance_type", "provenance must be an object"),
        ("dataset_schema", "unsupported modeling dataset"),
        ("feature_hash", "feature artifact SHA-256"),
        ("manifest_hash", "split manifest SHA-256"),
        ("model_hash", "model bundle SHA-256"),
        ("model_schema", "unsupported prototype model bundle"),
        ("policy_type", "evaluation policy must be an object"),
        ("policy", "evaluation policy is not trusted"),
        ("missing", "invalid prototype OOD calibration artifact"),
    ],
)
def test_rejects_invalid_artifact_envelope(tmp_path: Path, change: str, message: str) -> None:
    payload: object = valid_payload()
    if change == "root":
        payload = []
    else:
        root = cast("dict[str, object]", payload)
        provenance = nested(root, "provenance")
        if change == "schema":
            root["schema_version"] = "future"
        elif change == "calibration_schema":
            root["calibration_schema_version"] = "future"
        elif change == "provenance_type":
            root["provenance"] = []
        elif change == "dataset_schema":
            provenance["modeling_dataset_schema_version"] = "future"
        elif change == "feature_hash":
            nested(provenance, "feature_artifact")["sha256"] = "d" * 64
        elif change == "manifest_hash":
            nested(provenance, "split_manifest")["sha256"] = "d" * 64
        elif change == "model_hash":
            nested(provenance, "model_bundle")["sha256"] = "d" * 64
        elif change == "model_schema":
            nested(provenance, "model_bundle")["schema_version"] = "future"
        elif change == "policy_type":
            root["evaluation_policy"] = []
        elif change == "policy":
            nested(root, "evaluation_policy")["test_evaluated"] = True
        else:
            del root["geometry"]

    with pytest.raises(PrototypeCalibrationBundleError, match=message):
        load(tmp_path / "calibration.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("metadata_type", "calibration metadata must be an object"),
        ("schema", "metadata schema version"),
        ("model", "references a different model"),
        ("categories", "category order"),
        ("config_type", "configuration must be an object"),
        ("config", "configuration is not trusted"),
        ("data_type", "data policy must be an object"),
        ("data", "data policy is not trusted"),
        ("windows_type", "window count must be an integer"),
        ("windows_small", "window count is too small"),
        ("counts_type", "examples per class must be an object"),
        ("counts_keys", "counts must contain every category"),
        ("count_type", "calibration count must be an integer"),
        ("count_small", "counts do not match window count"),
        ("count_sum", "counts do not match window count"),
    ],
)
def test_rejects_invalid_calibration_metadata(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    payload = valid_payload()
    calibration = nested(payload, "calibration")
    data = nested(calibration, "data")
    counts = nested(calibration, "calibration_examples_per_class")
    if change == "metadata_type":
        payload["calibration"] = []
    elif change == "schema":
        calibration["schema_version"] = "future"
    elif change == "model":
        calibration["model_bundle_sha256"] = "d" * 64
    elif change == "categories":
        calibration["category_order"] = []
    elif change == "config_type":
        calibration["configuration"] = []
    elif change == "config":
        nested(calibration, "configuration")["bandwidth_method"] = "custom"
    elif change == "data_type":
        calibration["data"] = []
    elif change == "data":
        data["test_evaluated"] = True
    elif change == "windows_type":
        data["calibration_windows"] = True
    elif change == "windows_small":
        data["calibration_windows"] = 5
    elif change == "counts_type":
        calibration["calibration_examples_per_class"] = []
    elif change == "counts_keys":
        counts.pop(CATEGORY_LABELS[0])
    elif change == "count_type":
        counts[CATEGORY_LABELS[0]] = True
    elif change == "count_small":
        counts[CATEGORY_LABELS[0]] = 1
    else:
        counts[CATEGORY_LABELS[0]] = 3

    with pytest.raises(PrototypeCalibrationBundleError, match=message):
        load(tmp_path / "calibration.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("geometry_type", "geometry must be an object"),
        ("dtype", "must use float32"),
        ("shape_type", "shape must be an array"),
        ("shape_integer", "only integer values"),
        ("shape", "expected shape"),
        ("values_shape", "values do not match"),
        ("nonfinite", "finite values"),
        ("class_asymmetric", "class covariances must be symmetric"),
        ("global_asymmetric", "global covariance must be symmetric"),
        ("class_negative", "class covariance diagonals"),
        ("global_negative", "global covariance diagonal"),
    ],
)
def test_rejects_invalid_geometry(tmp_path: Path, change: str, message: str) -> None:
    payload = valid_payload()
    geometry = nested(payload, "geometry")
    means = nested(geometry, "class_means")
    if change == "geometry_type":
        payload["geometry"] = []
    elif change == "dtype":
        means["dtype"] = "float64"
    elif change == "shape_type":
        means["shape"] = 1
    elif change == "shape_integer":
        means["shape"] = [True]
    elif change == "shape":
        means["shape"] = [1]
    elif change == "values_shape":
        means["values"] = []
    elif change == "nonfinite":
        values = cast("list[list[float]]", means["values"])
        values[0][0] = float("nan")
    elif change in {"class_asymmetric", "class_negative"}:
        covariance = nested(geometry, "class_covariances")
        class_values = cast("list[list[list[float]]]", covariance["values"])
        if change == "class_asymmetric":
            class_values[0][0][1] = 1.0
        else:
            class_values[0][0][0] = -1.0
    else:
        covariance = nested(geometry, "global_covariance")
        global_values = cast("list[list[float]]", covariance["values"])
        if change == "global_asymmetric":
            global_values[0][1] = 1.0
        else:
            global_values[0][0] = -1.0

    with pytest.raises(PrototypeCalibrationBundleError, match=message):
        load(tmp_path / "calibration.json", payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("kdes_type", "class KDEs must be an object"),
        ("keys", "must contain every traffic category"),
        ("kde_type", "KDE must be an object"),
        ("method", "bandwidth method is not trusted"),
        ("count_type", "sample count must be an integer"),
        ("count", "sample count does not match"),
        ("samples_type", "samples must be an array"),
        ("samples_length", "must contain 2 numeric values"),
        ("samples_bool", "must contain 2 numeric values"),
        ("samples_nonfinite", "finite values"),
        ("bandwidth_type", "bandwidth must be numeric"),
        ("bandwidth_value", "finite and positive"),
        ("bandwidth_reproduction", "does not reproduce"),
    ],
)
def test_rejects_invalid_kdes(tmp_path: Path, change: str, message: str) -> None:
    payload = valid_payload()
    kdes = nested(payload, "class_kdes")
    category = CATEGORY_LABELS[0]
    kde = nested(kdes, category)
    if change == "kdes_type":
        payload["class_kdes"] = []
    elif change == "keys":
        kdes.pop(category)
    elif change == "kde_type":
        kdes[category] = []
    elif change == "method":
        kde["bandwidth_method"] = "custom"
    elif change == "count_type":
        kde["sample_count"] = True
    elif change == "count":
        kde["sample_count"] = 3
    elif change == "samples_type":
        kde["relative_distance_samples"] = 1
    elif change == "samples_length":
        kde["relative_distance_samples"] = []
    elif change == "samples_bool":
        kde["relative_distance_samples"] = [True, False]
    elif change == "samples_nonfinite":
        kde["relative_distance_samples"] = [0.0, float("inf")]
    elif change == "bandwidth_type":
        kde["bandwidth"] = True
    elif change == "bandwidth_value":
        kde["bandwidth"] = 0.0
    else:
        kde["bandwidth"] = 1.0

    with pytest.raises(PrototypeCalibrationBundleError, match=message):
        load(tmp_path / "calibration.json", payload)


def test_wraps_json_decode_failure(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{", encoding="utf-8")
    digest = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(PrototypeCalibrationBundleError, match="invalid prototype OOD"):
        load_prototype_ood_calibration(
            path,
            expected_sha256=digest,
            expected_model_bundle_sha256=MODEL_SHA256,
            expected_feature_artifact_sha256=FEATURE_SHA256,
            expected_split_manifest_sha256=MANIFEST_SHA256,
        )
