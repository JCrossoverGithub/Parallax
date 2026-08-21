"""Tests for manifest-bound modeling dataset ingestion."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import parallax.modeling.dataset as modeling_dataset
from parallax.data import DatasetPartition
from parallax.features import (
    CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    FEATURE_PARQUET_SCHEMA,
)
from parallax.modeling import ModelingDatasetError, load_partitioned_feature_dataset


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def feature_record(
    capture_id: str,
    vpn_status: str,
    application: str,
    category: str,
    value: float,
) -> dict[str, object]:
    record: dict[str, object] = {
        "window_id": f"{capture_id}:{value}",
        "capture_id": capture_id,
        "flow_id": "flow",
        "window_index": int(value),
        "vpn_status": vpn_status,
        "application": application,
        "category": category,
        "packet_count": 20,
    }
    record.update(dict.fromkeys(FEATURE_COLUMNS, value))
    return record


def base_records() -> list[dict[str, object]]:
    return [
        feature_record("train.pcap", "vpn", "ssh", "C2", 1.0),
        feature_record("validation.pcap", "nonvpn", "voip", "VOIP", 2.0),
        feature_record("calibration.pcap", "vpn", "youtube", "STREAMING", 3.0),
        feature_record("test.pcap", "nonvpn", "sftp", "FILE_TRANSFER", 4.0),
    ]


def write_features(path: Path, records: list[dict[str, object]] | None = None) -> None:
    table = pa.Table.from_pylist(
        records if records is not None else base_records(), schema=FEATURE_PARQUET_SCHEMA
    )
    pq.write_table(table, path, row_group_size=2)


def assignment(
    capture_id: str,
    partition: str,
    vpn_status: str,
    application: str,
    category: str,
    windows: int = 1,
) -> dict[str, object]:
    return {
        "capture_id": capture_id,
        "partition": partition,
        "vpn_status": vpn_status,
        "application": application,
        "category": category,
        "windows": windows,
    }


def manifest_payload(feature_path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(feature_path)
    assignments = [
        assignment("train.pcap", "train", "vpn", "ssh", "C2"),
        assignment("validation.pcap", "validation", "nonvpn", "voip", "VOIP"),
        assignment("calibration.pcap", "calibration", "vpn", "youtube", "STREAMING"),
        assignment("test.pcap", "test", "nonvpn", "sftp", "FILE_TRANSFER"),
    ]
    return {
        "schema_version": CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_schema_version": "vnat-capture-split-1",
        "source": {
            "file_name": feature_path.name,
            "format": "parquet",
            "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
            "file_size_bytes": feature_path.stat().st_size,
            "sha256": sha256_file(feature_path),
            "rows": parquet.metadata.num_rows,
            "columns": parquet.metadata.num_columns,
            "row_groups": parquet.metadata.num_row_groups,
        },
        "solver": {"optimality_proven": True},
        "summary": {
            "captures": 4,
            "windows": 4,
            "partitions": {
                partition: {"captures": 1, "windows": 1}
                for partition in ("train", "validation", "calibration", "test")
            },
        },
        "assignments": assignments,
    }


def write_manifest(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def load_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    write_features(features)
    payload = manifest_payload(features)
    return features, manifest, payload


def remove_validation_partition(payload: dict[str, Any]) -> dict[str, Any]:
    payload["assignments"][1]["partition"] = "train"
    payload["summary"]["partitions"]["train"] = {"captures": 2, "windows": 2}
    payload["summary"]["partitions"]["validation"] = {"captures": 0, "windows": 0}
    return payload


def test_loads_read_only_manifest_bound_partitions(tmp_path: Path) -> None:
    features, manifest, payload = load_fixture(tmp_path)
    manifest_sha256 = write_manifest(manifest, payload)

    dataset = load_partitioned_feature_dataset(
        features,
        manifest,
        expected_manifest_sha256=manifest_sha256,
    )

    assert dataset.feature_artifact_sha256 == sha256_file(features)
    assert dataset.split_manifest_sha256 == manifest_sha256
    assert tuple(item.partition for item in dataset.partitions) == tuple(DatasetPartition)

    training = dataset.partition(DatasetPartition.TRAIN)
    assert training.windows == 1
    assert training.captures == 1
    assert training.features.shape == (1, 129)
    assert training.features.dtype == np.float32
    assert np.all(training.features == 1.0)
    assert training.categories.tolist() == ["C2"]
    assert training.capture_ids.tolist() == ["train.pcap"]
    assert training.vpn_statuses.tolist() == ["vpn"]
    assert training.applications.tolist() == ["ssh"]

    for partition in dataset.partitions:
        for array in (
            partition.features,
            partition.categories,
            partition.capture_ids,
            partition.vpn_statuses,
            partition.applications,
        ):
            assert not array.flags.writeable

    with pytest.raises(KeyError, match="train"):
        dataset.partition("train")  # type: ignore[arg-type]


def test_partition_scan_skips_roles_absent_from_individual_batches(tmp_path: Path) -> None:
    features, manifest, payload = load_fixture(tmp_path)
    write_manifest(manifest, payload)
    verified = modeling_dataset._read_manifest(manifest)
    one_row_batches = pq.read_table(features).to_batches(max_chunksize=1)
    parquet = SimpleNamespace(iter_batches=lambda **kwargs: iter(one_row_batches))

    partitions = modeling_dataset._scan_partitions(
        parquet,  # type: ignore[arg-type]
        verified.assignments,
    )

    assert [partition.windows for partition in partitions] == [1, 1, 1, 1]


def test_rejects_manifest_checksum_before_json_parse(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(ModelingDatasetError, match="manifest SHA-256 mismatch"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: [], "root must be an object"),
        (lambda payload: payload.update(schema_version="wrong") or payload, "manifest schema"),
        (
            lambda payload: payload.update(split_schema_version="wrong") or payload,
            "capture split schema",
        ),
        (
            lambda payload: payload["source"].update(format="csv") or payload,
            "source format",
        ),
        (
            lambda payload: payload["source"].update(schema_version="wrong") or payload,
            "unsupported feature schema",
        ),
        (
            lambda payload: payload["source"].update(rows=0) or payload,
            "source counts must be positive",
        ),
        (
            lambda payload: payload["solver"].update(optimality_proven=False) or payload,
            "proven optimal solution",
        ),
        (
            lambda payload: payload.update(assignments=[]) or payload,
            "must contain capture assignments",
        ),
        (
            lambda payload: payload["assignments"].append(payload["assignments"][0]) or payload,
            "assignments must be unique",
        ),
        (
            remove_validation_partition,
            "every modeling partition",
        ),
        (
            lambda payload: payload["assignments"][0].update(capture_id="") or payload,
            "identifiers must not be empty",
        ),
        (
            lambda payload: payload["assignments"][0].update(windows=0) or payload,
            "window counts must be positive",
        ),
        (
            lambda payload: payload["assignments"][0].update(category="CHAT") or payload,
            "labels do not agree",
        ),
        (
            lambda payload: payload["summary"].update(captures=3) or payload,
            "capture summary",
        ),
        (
            lambda payload: payload["summary"].update(windows=3) or payload,
            "window summary",
        ),
        (
            lambda payload: payload["summary"]["partitions"]["train"].update(captures=2) or payload,
            "partition capture summary",
        ),
        (
            lambda payload: payload["summary"]["partitions"]["train"].update(windows=2) or payload,
            "partition window summary",
        ),
    ],
)
def test_rejects_invalid_manifest_contract(
    tmp_path: Path,
    change: Any,
    message: str,
) -> None:
    features, manifest, original = load_fixture(tmp_path)
    payload = change(deepcopy(original))
    expected_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match=message):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=expected_sha256,
        )


def test_wraps_malformed_json_error(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    manifest.write_text("{", encoding="utf-8")

    with pytest.raises(ModelingDatasetError, match="invalid split manifest"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=sha256_file(manifest),
        )


def test_rejects_feature_size_mismatch_before_hashing(tmp_path: Path) -> None:
    features, manifest, payload = load_fixture(tmp_path)
    payload["source"]["file_size_bytes"] = features.stat().st_size + 1
    expected_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match="artifact size"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=expected_sha256,
        )


def test_rejects_feature_checksum_before_parquet_load(tmp_path: Path) -> None:
    features = tmp_path / "not-parquet.parquet"
    manifest = tmp_path / "capture-splits.json"
    features.write_bytes(b"untrusted")
    payload = {
        "schema_version": CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_schema_version": "vnat-capture-split-1",
        "source": {
            "format": "parquet",
            "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
            "file_size_bytes": len(b"untrusted"),
            "sha256": "0" * 64,
            "rows": 4,
            "columns": 1,
            "row_groups": 1,
        },
        "solver": {"optimality_proven": True},
        "summary": {
            "captures": 4,
            "windows": 4,
            "partitions": {
                partition: {"captures": 1, "windows": 1}
                for partition in ("train", "validation", "calibration", "test")
            },
        },
        "assignments": [
            assignment("train.pcap", "train", "vpn", "ssh", "C2"),
            assignment("validation.pcap", "validation", "nonvpn", "voip", "VOIP"),
            assignment("calibration.pcap", "calibration", "vpn", "youtube", "STREAMING"),
            assignment("test.pcap", "test", "nonvpn", "sftp", "FILE_TRANSFER"),
        ],
    }
    manifest_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match="artifact SHA-256"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def test_rejects_wrong_feature_schema(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    pq.write_table(pa.table({"value": [1]}), features)
    payload = manifest_payload_for_arbitrary_source(features)
    manifest_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match="versioned feature schema"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def manifest_payload_for_arbitrary_source(feature_path: Path) -> dict[str, Any]:
    payload = manifest_payload_stub()
    payload["source"].update(
        file_size_bytes=feature_path.stat().st_size,
        sha256=sha256_file(feature_path),
    )
    return payload


def manifest_payload_stub() -> dict[str, Any]:
    return {
        "schema_version": CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_schema_version": "vnat-capture-split-1",
        "source": {
            "format": "parquet",
            "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
            "file_size_bytes": 1,
            "sha256": "0" * 64,
            "rows": 4,
            "columns": 137,
            "row_groups": 1,
        },
        "solver": {"optimality_proven": True},
        "summary": {
            "captures": 4,
            "windows": 4,
            "partitions": {
                partition: {"captures": 1, "windows": 1}
                for partition in ("train", "validation", "calibration", "test")
            },
        },
        "assignments": [
            assignment("train.pcap", "train", "vpn", "ssh", "C2"),
            assignment("validation.pcap", "validation", "nonvpn", "voip", "VOIP"),
            assignment("calibration.pcap", "calibration", "vpn", "youtube", "STREAMING"),
            assignment("test.pcap", "test", "nonvpn", "sftp", "FILE_TRANSFER"),
        ],
    }


def test_rejects_parquet_metadata_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features, manifest, payload = load_fixture(tmp_path)
    manifest_sha256 = write_manifest(manifest, payload)
    original = modeling_dataset.pq.ParquetFile  # type: ignore[attr-defined]

    def mismatched(path: Path) -> SimpleNamespace:
        parquet = original(path)
        metadata = SimpleNamespace(
            num_rows=parquet.metadata.num_rows,
            num_columns=parquet.metadata.num_columns,
            num_row_groups=parquet.metadata.num_row_groups + 1,
        )
        return SimpleNamespace(schema_arrow=parquet.schema_arrow, metadata=metadata)

    monkeypatch.setattr(
        modeling_dataset.pq,  # type: ignore[attr-defined]
        "ParquetFile",
        mismatched,
    )

    with pytest.raises(ModelingDatasetError, match="metadata does not match"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda records: records.__setitem__(
                0,
                feature_record("unknown.pcap", "vpn", "ssh", "C2", 1.0),
            ),
            "unassigned capture",
        ),
        (
            lambda records: records[0].update(vpn_status="nonvpn"),
            "labels do not match",
        ),
        (
            lambda records: records[0].update({FEATURE_COLUMNS[0]: float("nan")}),
            "non-finite",
        ),
    ],
)
def test_rejects_feature_content_mismatch(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    records = base_records()
    payload_assignments = manifest_payload_for_records(records)
    mutation(records)
    write_features(features, records)
    payload = payload_assignments(features)
    manifest_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match=message):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def manifest_payload_for_records(records: list[dict[str, object]]) -> Any:
    identities = [
        (
            str(record["capture_id"]),
            str(record["vpn_status"]),
            str(record["application"]),
            str(record["category"]),
        )
        for record in records
    ]

    def build(feature_path: Path) -> dict[str, Any]:
        payload = manifest_payload(feature_path)
        payload["assignments"] = [
            assignment(capture_id, partition, vpn_status, application, category)
            for (capture_id, vpn_status, application, category), partition in zip(
                identities,
                ("train", "validation", "calibration", "test"),
                strict=True,
            )
        ]
        return payload

    return build


def test_rejects_feature_window_count_mismatch(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "capture-splits.json"
    records = [
        feature_record("train.pcap", "vpn", "ssh", "C2", 1.0),
        feature_record("train.pcap", "vpn", "ssh", "C2", 1.1),
        feature_record("validation.pcap", "nonvpn", "voip", "VOIP", 2.0),
        feature_record("validation.pcap", "nonvpn", "voip", "VOIP", 2.1),
        feature_record("calibration.pcap", "vpn", "youtube", "STREAMING", 3.0),
        feature_record("calibration.pcap", "vpn", "youtube", "STREAMING", 3.1),
        feature_record("test.pcap", "nonvpn", "sftp", "FILE_TRANSFER", 4.0),
        feature_record("test.pcap", "nonvpn", "sftp", "FILE_TRANSFER", 4.1),
    ]
    write_features(features, records)
    payload = manifest_payload(features)
    for item in payload["assignments"]:
        item["windows"] = 2
    payload["assignments"][0]["windows"] = 3
    payload["assignments"][1]["windows"] = 1
    payload["summary"]["windows"] = 8
    for summary in payload["summary"]["partitions"].values():
        summary["windows"] = 2
    payload["summary"]["partitions"]["train"]["windows"] = 3
    payload["summary"]["partitions"]["validation"]["windows"] = 1
    manifest_sha256 = write_manifest(manifest, payload)

    with pytest.raises(ModelingDatasetError, match="window counts do not match"):
        load_partitioned_feature_dataset(
            features,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )
