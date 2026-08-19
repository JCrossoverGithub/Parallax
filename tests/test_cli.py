"""Tests for the Parallax command-line entry point."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from warnings import catch_warnings, simplefilter

import pandas as pd
from pytest import CaptureFixture, MonkeyPatch

from parallax import __version__
from parallax.cli import main


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
