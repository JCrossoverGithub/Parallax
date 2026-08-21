"""Tests for immutable capture-split manifest publication."""

import json
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import parallax.features.split_manifest as split_manifest
from parallax.data import (
    APPLICATION_TO_CATEGORY,
    Application,
    CaptureGroup,
    CaptureSplitConfig,
    CaptureSplitResult,
    DatasetPartition,
    TrafficCategory,
    VpnStatus,
)
from parallax.features import (
    CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    FEATURE_PARQUET_SCHEMA,
    CaptureSplitManifestError,
    export_capture_split_manifest,
)


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def capture_labels() -> list[tuple[str, VpnStatus, Application, TrafficCategory]]:
    labels: list[tuple[str, VpnStatus, Application, TrafficCategory]] = []
    for category in TrafficCategory:
        applications = [
            application
            for application, mapped_category in APPLICATION_TO_CATEGORY.items()
            if mapped_category is category
        ]
        for index in range(8):
            application = applications[index % len(applications)]
            vpn_status = VpnStatus.VPN if index % 2 else VpnStatus.NON_VPN
            labels.append(
                (
                    f"{vpn_status.value}_{application.value}_capture{index}.pcap",
                    vpn_status,
                    application,
                    category,
                )
            )
    return labels


def feature_record(
    capture_id: str,
    vpn_status: str,
    application: str,
    category: str,
    window_index: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "window_id": f"{capture_id}:flow:{window_index}",
        "capture_id": capture_id,
        "flow_id": "flow",
        "window_index": window_index,
        "vpn_status": vpn_status,
        "application": application,
        "category": category,
        "packet_count": 20,
    }
    record.update(dict.fromkeys(FEATURE_COLUMNS, 0.0))
    return record


def write_feature_artifact(
    path: Path,
    *,
    rows_per_capture: int = 20,
    labels: list[tuple[str, VpnStatus, Application, TrafficCategory]] | None = None,
) -> None:
    selected = labels if labels is not None else capture_labels()
    records = [
        feature_record(
            capture_id,
            vpn_status.value,
            application.value,
            category.value,
            window_index,
        )
        for capture_id, vpn_status, application, category in selected
        for window_index in range(rows_per_capture)
    ]
    pq.write_table(
        pa.Table.from_pylist(records, schema=FEATURE_PARQUET_SCHEMA),
        path,
        row_group_size=17,
    )


def test_exports_deterministic_optimal_split_manifest(tmp_path: Path) -> None:
    source = tmp_path / "features.parquet"
    output = tmp_path / "splits" / "capture-splits.json"
    write_feature_artifact(source)

    report = export_capture_split_manifest(
        source,
        output,
        expected_sha256=sha256_file(source),
    )

    encoded = output.read_bytes()
    payload = json.loads(encoded)
    expected_assignments = sorted(
        payload["assignments"],
        key=lambda assignment: assignment["capture_id"],
    )

    assert payload == report.as_dict()
    assert encoded == (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    assert payload["schema_version"] == CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION
    assert payload["source"] == {
        "columns": 137,
        "file_name": "features.parquet",
        "file_size_bytes": source.stat().st_size,
        "format": "parquet",
        "row_groups": 48,
        "rows": 800,
        "schema_version": "vnat-feature-artifact-1",
        "sha256": sha256_file(source),
    }
    assert payload["solver"]["optimality_proven"] is True
    assert payload["assignments"] == expected_assignments
    assert payload["summary"]["captures"] == 40
    assert payload["summary"]["windows"] == 800
    assert set(payload["summary"]["partitions"]) == {
        partition.value for partition in DatasetPartition
    }
    assert report.output_file_size_bytes == len(encoded)
    assert report.output_sha256 == sha256(encoded).hexdigest()
    assert report.source_rows == 800
    assert report.source_columns == 137
    assert report.source_row_groups == 48

    for summary in payload["summary"]["partitions"].values():
        assert summary["captures"] > 0
        assert summary["windows"] > 0
        assert summary["window_fraction"] > 0.0
        assert summary["captures_by_category"]
        assert summary["windows_by_category"]
        assert summary["captures_by_vpn_status"]
        assert summary["windows_by_vpn_status"]
        assert summary["captures_by_application"]
        assert summary["windows_by_application"]


@pytest.mark.parametrize("suffix", ["", ".parquet"])
def test_rejects_non_json_output(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(CaptureSplitManifestError, match=r"must use the \.json extension"):
        export_capture_split_manifest(
            tmp_path / "missing.parquet",
            tmp_path / f"splits{suffix}",
            expected_sha256="0" * 64,
        )


def test_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "splits.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(CaptureSplitManifestError, match="already exists"):
        export_capture_split_manifest(
            tmp_path / "missing.parquet",
            output,
            expected_sha256="0" * 64,
        )


def test_rejects_checksum_mismatch_before_parquet_load(tmp_path: Path) -> None:
    source = tmp_path / "not-parquet.parquet"
    source.write_bytes(b"untrusted")

    with pytest.raises(CaptureSplitManifestError, match="SHA-256 mismatch"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256="0" * 64,
        )


def test_rejects_wrong_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"value": [1]}), source)

    with pytest.raises(CaptureSplitManifestError, match="versioned feature schema"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_rejects_empty_feature_artifact(tmp_path: Path) -> None:
    source = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=FEATURE_PARQUET_SCHEMA), source)

    with pytest.raises(CaptureSplitManifestError, match="at least one window"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_rejects_invalid_feature_label(tmp_path: Path) -> None:
    source = tmp_path / "invalid-label.parquet"
    record = feature_record("capture.pcap", "unknown", "voip", "VOIP", 0)
    pq.write_table(pa.Table.from_pylist([record], schema=FEATURE_PARQUET_SCHEMA), source)

    with pytest.raises(CaptureSplitManifestError, match="invalid label"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_rejects_inconsistent_capture_labels(tmp_path: Path) -> None:
    source = tmp_path / "inconsistent.parquet"
    records = [
        feature_record("capture.pcap", "vpn", "voip", "VOIP", 0),
        feature_record("capture.pcap", "nonvpn", "voip", "VOIP", 1),
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=FEATURE_PARQUET_SCHEMA), source)

    with pytest.raises(CaptureSplitManifestError, match="inconsistent labels"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_rejects_application_category_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "mismatch.parquet"
    record = feature_record("capture.pcap", "vpn", "voip", "CHAT", 0)
    pq.write_table(pa.Table.from_pylist([record], schema=FEATURE_PARQUET_SCHEMA), source)

    with pytest.raises(CaptureSplitManifestError, match="application/category mismatch"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_rejects_incomplete_feature_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "features.parquet"
    write_feature_artifact(source, rows_per_capture=1)
    original = split_manifest.pq.ParquetFile  # type: ignore[attr-defined]

    def incomplete_parquet(path: Path) -> SimpleNamespace:
        parquet = original(path)
        first = next(parquet.iter_batches(columns=list(split_manifest._LABEL_COLUMNS)))
        return SimpleNamespace(
            schema_arrow=parquet.schema_arrow,
            metadata=parquet.metadata,
            iter_batches=lambda **kwargs: iter([first.slice(0, 1)]),
        )

    monkeypatch.setattr(
        split_manifest.pq,  # type: ignore[attr-defined]
        "ParquetFile",
        incomplete_parquet,
    )

    with pytest.raises(CaptureSplitManifestError, match="does not match Parquet metadata"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
        )


def test_refuses_feasible_result_without_proven_optimality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "features.parquet"
    write_feature_artifact(source, rows_per_capture=1)
    optimal = split_manifest.assign_capture_splits  # type: ignore[attr-defined]

    def time_limited(
        captures: Iterable[CaptureGroup],
        *,
        config: CaptureSplitConfig | None = None,
    ) -> CaptureSplitResult:
        result = optimal(captures, config=config)
        return replace(result, optimal=False, solver_message="time limit reached")

    monkeypatch.setattr(split_manifest, "assign_capture_splits", time_limited)

    with pytest.raises(CaptureSplitManifestError, match="without proven optimality"):
        export_capture_split_manifest(
            source,
            tmp_path / "splits.json",
            expected_sha256=sha256_file(source),
            config=replace(CaptureSplitConfig(), minimum_category_windows=1),
        )

    assert not (tmp_path / "splits.json").exists()
