"""Tests for the Parallax command-line entry point."""

import json
import sys
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

    main(["dataset", "inspect", str(path)])

    output = json.loads(capsys.readouterr().out)
    assert output["source"] == str(path)
    assert output["summary"]["connections"] == 1
