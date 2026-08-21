"""Tests for the Parallax command-line entry point."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from warnings import catch_warnings, simplefilter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pytest import CaptureFixture, MonkeyPatch

from parallax import __version__
from parallax.cli import main
from parallax.data import WINDOW_PARQUET_SCHEMA


def test_package_version_matches_initial_release() -> None:
    assert __version__ == "0.1.0"


def test_main_reports_name_and_version(
    capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["parallax"])
    main()

    assert capsys.readouterr().out == "Parallax 0.1.0\n"


def test_main_inspects_vnat_dataset(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "raw.h5"
    dataframe = pd.DataFrame(
        {
            "connection": [("source", 1000, "destination", 443, 6)],
            "timestamps": [[1.0]],
            "sizes": [[100]],
            "directions": [[0]],
            "file_names": ["vpn_youtube_capture1.pcap"],
        }
    )
    with catch_warnings():
        simplefilter("ignore", pd.errors.PerformanceWarning)
        dataframe.to_hdf(path, key="data")

    expected_sha256 = sha256(path.read_bytes()).hexdigest()
    main(
        [
            "dataset",
            "inspect",
            str(path),
            "--expected-sha256",
            expected_sha256,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["source"] == str(path)
    assert output["summary"]["connections"] == 1


def test_main_exports_vnat_windows(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    source = tmp_path / "raw.h5"
    output = tmp_path / "processed" / "windows.parquet"
    packet_count = 20
    dataframe = pd.DataFrame(
        {
            "connection": [("source", 1000, "destination", 443, 6)],
            "timestamps": [[index * 0.01 for index in range(packet_count)]],
            "sizes": [[100] * packet_count],
            "directions": [[index % 2 for index in range(packet_count)]],
            "file_names": ["vpn_youtube_capture1.pcap"],
        }
    )
    with catch_warnings():
        simplefilter("ignore", pd.errors.PerformanceWarning)
        dataframe.to_hdf(source, key="data")

    expected_sha256 = sha256(source.read_bytes()).hexdigest()
    main(
        [
            "dataset",
            "extract-windows",
            str(source),
            str(output),
            "--expected-sha256",
            expected_sha256,
            "--threshold-policy",
            "paper-literal",
            "--window-seconds",
            "20.48",
            "--minimum-packets",
            "20",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["output"]["path"] == str(output)
    assert report["configuration"] == {
        "eligibility_operator": ">=",
        "minimum_packets": 20,
        "threshold_policy": "paper-literal",
        "window_seconds": 20.48,
    }
    assert report["summary"]["windows"] == 1
    assert output.is_file()
    assert output.with_suffix(".manifest.json").is_file()


def test_main_exports_vnat_features(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    source = tmp_path / "windows.parquet"
    output = tmp_path / "processed" / "features.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "window_id": "capture:flow:0",
                "capture_id": "capture.pcap",
                "flow_id": "flow",
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
            }
        ],
        schema=WINDOW_PARQUET_SCHEMA,
    )
    pq.write_table(table, source)
    expected_sha256 = sha256(source.read_bytes()).hexdigest()

    main(
        [
            "dataset",
            "extract-features",
            str(source),
            str(output),
            "--expected-sha256",
            expected_sha256,
            "--byte-total-policy",
            "corrected",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["output"]["path"] == str(output)
    assert report["configuration"] == {
        "byte_total_policy": "corrected",
        "time_bin_seconds": 0.01,
        "window_seconds": 40.96,
    }
    assert report["summary"]["windows"] == 1
    assert output.is_file()
    assert output.with_suffix(".manifest.json").is_file()


def test_main_exports_capture_split_manifest(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "features.parquet"
    output = tmp_path / "capture-splits.json"
    expected_sha256 = "1" * 64
    expected_payload = {
        "schema_version": "vnat-capture-split-manifest-1",
        "summary": {"captures": 162, "windows": 15_095},
    }

    def fake_export(
        selected_source: str,
        selected_output: str,
        *,
        expected_sha256: str,
    ) -> SimpleNamespace:
        assert selected_source == str(source)
        assert selected_output == str(output)
        assert expected_sha256 == "1" * 64
        return SimpleNamespace(as_dict=lambda: expected_payload)

    monkeypatch.setattr("parallax.cli.export_capture_split_manifest", fake_export)

    main(
        [
            "dataset",
            "split-features",
            str(source),
            str(output),
            "--expected-sha256",
            expected_sha256,
        ]
    )

    assert json.loads(capsys.readouterr().out) == expected_payload
