import hashlib
import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parallax.data import WINDOW_PARQUET_SCHEMA
from parallax.features import (
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    FEATURE_PARQUET_SCHEMA,
    FeatureExportError,
    calculate_feature_vector,
    export_vnat_features,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_records() -> list[dict[str, object]]:
    return [
        {
            "window_id": "capture-a:flow-a:0",
            "capture_id": "capture-a.pcap",
            "flow_id": "flow-a",
            "window_index": 0,
            "start_offset_seconds": 0.0,
            "end_offset_seconds": 40.96,
            "vpn_status": "vpn",
            "application": "voip",
            "category": "VOIP",
            "packet_count": 3,
            "timestamps": [0.0, 0.01, 6.0],
            "sizes": [100, 200, 300],
            "directions": [1, 0, 1],
        },
        {
            "window_id": "capture-b:flow-b:1",
            "capture_id": "capture-b.pcap",
            "flow_id": "flow-b",
            "window_index": 1,
            "start_offset_seconds": 40.96,
            "end_offset_seconds": 81.92,
            "vpn_status": "nonvpn",
            "application": "ssh",
            "category": "C2",
            "packet_count": 2,
            "timestamps": [1.0, 2.0],
            "sizes": [400, 500],
            "directions": [0, 1],
        },
    ]


def write_window_artifact(path: Path, records: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(records, schema=WINDOW_PARQUET_SCHEMA)
    pq.write_table(table, path, row_group_size=1)


def test_exports_versioned_feature_artifact_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "windows.parquet"
    output = tmp_path / "features.parquet"
    records = window_records()
    write_window_artifact(source, records)

    report = export_vnat_features(
        source,
        output,
        expected_sha256=sha256_file(source),
    )

    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(output)
    exported = parquet.read().to_pylist()
    expected_vector = calculate_feature_vector(
        cast("list[float]", records[0]["timestamps"]),
        cast("list[int]", records[0]["sizes"]),
        cast("list[int]", records[0]["directions"]),
    )

    assert parquet.schema_arrow.equals(FEATURE_PARQUET_SCHEMA, check_metadata=True)
    assert parquet.num_row_groups == 2
    assert len(exported) == 2
    assert exported[0]["window_id"] == records[0]["window_id"]
    assert [exported[0][name] for name in FEATURE_COLUMNS] == pytest.approx(expected_vector)
    assert report.summary.windows == 2
    assert report.summary.packets == 5
    assert report.summary.captures == 2
    assert report.summary.windows_by_vpn_status == {"nonvpn": 1, "vpn": 1}
    assert report.summary.windows_by_application == {"ssh": 1, "voip": 1}
    assert report.summary.windows_by_category == {"C2": 1, "VOIP": 1}
    assert report.output_sha256 == sha256_file(output)
    assert manifest == report.as_dict()
    assert manifest["schema_version"] == FEATURE_ARTIFACT_SCHEMA_VERSION
    assert manifest["source"]["schema_version"] == "vnat-window-1"
    assert manifest["configuration"]["byte_total_policy"] == "release-compatible"


@pytest.mark.parametrize("suffix", ["", ".csv"])
def test_rejects_non_parquet_output(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(FeatureExportError, match=r"must use the \.parquet extension"):
        export_vnat_features(
            tmp_path / "missing.parquet",
            tmp_path / f"features{suffix}",
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize("existing", ["output", "manifest"])
def test_refuses_existing_destination(tmp_path: Path, existing: str) -> None:
    output = tmp_path / "features.parquet"
    target = output if existing == "output" else output.with_suffix(".manifest.json")
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FeatureExportError, match="already exists"):
        export_vnat_features(
            tmp_path / "missing.parquet",
            output,
            expected_sha256="0" * 64,
        )


def test_rejects_checksum_mismatch_before_parquet_load(tmp_path: Path) -> None:
    source = tmp_path / "not-parquet.parquet"
    source.write_bytes(b"untrusted")

    with pytest.raises(FeatureExportError, match="SHA-256 mismatch"):
        export_vnat_features(
            source,
            tmp_path / "features.parquet",
            expected_sha256="0" * 64,
        )


def test_rejects_wrong_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"value": [1]}), source)

    with pytest.raises(FeatureExportError, match="versioned window schema"):
        export_vnat_features(
            source,
            tmp_path / "features.parquet",
            expected_sha256=sha256_file(source),
        )


def test_rejects_inconsistent_packet_count_without_final_outputs(tmp_path: Path) -> None:
    source = tmp_path / "windows.parquet"
    output = tmp_path / "features.parquet"
    records = window_records()
    records[0]["packet_count"] = 4
    write_window_artifact(source, records)

    with pytest.raises(FeatureExportError, match="packet count"):
        export_vnat_features(
            source,
            output,
            expected_sha256=sha256_file(source),
        )

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
