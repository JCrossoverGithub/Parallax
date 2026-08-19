import json
from hashlib import sha256
from pathlib import Path
from typing import cast
from warnings import catch_warnings, simplefilter

import pandas as pd
import pyarrow.parquet as pq
import pytest

from parallax.data import (
    WINDOW_PARQUET_SCHEMA,
    WINDOW_SCHEMA_VERSION,
    VnatWindowError,
    WindowExportError,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    export_vnat_windows,
)


def _dataframe(*, invalid_direction: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "connection": [
                ("10.0.0.1", 1000, "10.0.0.2", 443, 6),
                ("10.0.0.3", 2000, "10.0.0.4", 22, 6),
            ],
            "timestamps": [
                [100.0 + index * 0.01 for index in range(21)],
                [200.0 + index * 0.01 for index in range(20)],
            ],
            "sizes": [list(range(100, 121)), list(range(200, 220))],
            "directions": [
                [2 if invalid_direction else index % 2 for index in range(21)],
                [index % 2 for index in range(20)],
            ],
            "file_names": [
                "nonvpn_youtube_capture1.pcap",
                "vpn_ssh_capture1.pcap",
            ],
        }
    )


def _write_hdf(path: Path, dataframe: pd.DataFrame) -> str:
    with catch_warnings():
        simplefilter("ignore", pd.errors.PerformanceWarning)
        dataframe.to_hdf(path, key="data")
    return sha256(path.read_bytes()).hexdigest()


def test_exports_release_compatible_windows_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "raw.h5"
    expected_sha256 = _write_hdf(source, _dataframe())
    output = tmp_path / "processed" / "windows.parquet"

    report = export_vnat_windows(
        source,
        output,
        expected_sha256=expected_sha256,
    )

    manifest = output.with_suffix(".manifest.json")
    assert output.is_file()
    assert manifest.is_file()
    assert report.source == str(source)
    assert report.source_sha256 == expected_sha256
    assert report.output == str(output)
    assert report.output_file_size_bytes == output.stat().st_size
    assert report.output_sha256 == sha256(output.read_bytes()).hexdigest()
    assert report.manifest == str(manifest)
    assert report.summary.source_connections == 2
    assert report.summary.source_packets == 41
    assert report.summary.source_captures == 2
    assert report.summary.represented_captures == 1
    assert report.summary.omitted_captures == ["vpn_ssh_capture1.pcap"]
    assert report.summary.windows == 1
    assert report.summary.packets == 21
    assert report.summary.windows_by_vpn_status == {"nonvpn": 1}
    assert report.summary.windows_by_application == {"youtube": 1}
    assert report.summary.windows_by_category == {"STREAMING": 1}

    table = pq.read_table(output)
    assert table.schema == WINDOW_PARQUET_SCHEMA
    assert table.num_rows == 1
    assert table.column_names == WINDOW_PARQUET_SCHEMA.names
    assert "connection" not in table.column_names
    assert table["capture_id"][0].as_py() == "nonvpn_youtube_capture1.pcap"
    assert table["packet_count"][0].as_py() == 21
    assert len(table["timestamps"][0].as_py()) == 21
    assert len(table["sizes"][0].as_py()) == 21
    assert len(table["directions"][0].as_py()) == 21
    assert table.schema.metadata == {b"parallax.schema_version": b"vnat-window-1"}

    serialized = json.loads(manifest.read_text(encoding="utf-8"))
    assert serialized == report.as_dict()
    assert serialized["schema_version"] == WINDOW_SCHEMA_VERSION
    assert serialized["configuration"]["eligibility_operator"] == ">"
    assert serialized["output"]["compression"] == "zstd"


def test_exports_paper_literal_threshold(tmp_path: Path) -> None:
    source = tmp_path / "raw.h5"
    expected_sha256 = _write_hdf(source, _dataframe())
    output = tmp_path / "windows.parquet"
    config = WindowExtractionConfig(
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )

    report = export_vnat_windows(
        source,
        output,
        expected_sha256=expected_sha256,
        config=config,
    )

    assert report.summary.windows == 2
    assert report.summary.represented_captures == 2
    assert report.summary.omitted_captures == []
    assert report.summary.windows_by_category == {"C2": 1, "STREAMING": 1}
    configuration = cast(dict[str, object], report.as_dict()["configuration"])
    assert configuration["eligibility_operator"] == ">="
    assert pq.read_table(output).num_rows == 2


def test_writes_a_full_batch_without_retaining_a_partial_batch(tmp_path: Path) -> None:
    source = tmp_path / "raw.h5"
    timestamps = [
        window_index * 100.0 + packet_index
        for window_index in range(64)
        for packet_index in range(21)
    ]
    dataframe = pd.DataFrame(
        {
            "connection": [("10.0.0.1", 1000, "10.0.0.2", 443, 6)],
            "timestamps": [timestamps],
            "sizes": [[100] * len(timestamps)],
            "directions": [[0] * len(timestamps)],
            "file_names": ["nonvpn_youtube_capture1.pcap"],
        }
    )
    expected_sha256 = _write_hdf(source, dataframe)
    output = tmp_path / "windows.parquet"

    report = export_vnat_windows(
        source,
        output,
        expected_sha256=expected_sha256,
        config=WindowExtractionConfig(window_seconds=100.0),
    )

    parquet = pq.ParquetFile(output)
    assert report.summary.windows == 64
    assert parquet.metadata.num_row_groups == 1


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("extension", "must use the .parquet extension"),
        ("output", "output already exists"),
        ("manifest", "manifest already exists"),
    ],
)
def test_rejects_unsafe_destination(
    tmp_path: Path,
    setup: str,
    message: str,
) -> None:
    source = tmp_path / "raw.h5"
    expected_sha256 = _write_hdf(source, _dataframe())
    output = tmp_path / ("windows.txt" if setup == "extension" else "windows.parquet")

    if setup == "output":
        output.touch()
    elif setup == "manifest":
        output.with_suffix(".manifest.json").touch()

    with pytest.raises(WindowExportError, match=message):
        export_vnat_windows(source, output, expected_sha256=expected_sha256)


def test_failed_extraction_leaves_no_processed_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "raw.h5"
    expected_sha256 = _write_hdf(source, _dataframe(invalid_direction=True))
    output = tmp_path / "processed" / "windows.parquet"

    with pytest.raises(VnatWindowError, match="directions must be zero or one"):
        export_vnat_windows(source, output, expected_sha256=expected_sha256)

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
