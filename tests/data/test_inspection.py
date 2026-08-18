from pathlib import Path
from typing import Any, cast
from warnings import catch_warnings, simplefilter

import pandas as pd
import pytest

from parallax.data import (
    RAW_COLUMNS,
    VnatDatasetError,
    inspect_raw_dataframe,
    inspect_vnat_file,
)


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "connection": [
                ("10.0.0.1", 1000, "10.0.0.2", 443, 6),
                ("10.0.0.3", 2000, "10.0.0.4", 443, 6),
                ("10.0.0.5", 3000, "10.0.0.6", 443, 6),
            ],
            "timestamps": [[1.0, 1.1], [2.0], [3.0, 3.1, 3.2]],
            "sizes": [[100, 200], [300], [400, 500, 600]],
            "directions": [[0, 1], [1], [0, 1, 0]],
            "file_names": pd.Series(
                [
                    "vpn_youtube_capture1.pcap",
                    "vpn_youtube_capture1.pcap",
                    "nonvpn_ssh_capture1.pcap",
                ],
                dtype=object,
            ),
        }
    )


def test_inspect_raw_dataframe_summarizes_valid_data() -> None:
    summary = inspect_raw_dataframe(_valid_dataframe())

    assert summary.connections == 3
    assert summary.packets == 6
    assert summary.captures == 2
    assert summary.minimum_packets_per_connection == 1
    assert summary.median_packets_per_connection == 2.0
    assert summary.maximum_packets_per_connection == 3
    assert summary.captures_by_vpn_status == {"nonvpn": 1, "vpn": 1}
    assert summary.captures_by_category == {"C2": 1, "STREAMING": 1}
    assert summary.captures_by_application == {"ssh": 1, "youtube": 1}
    assert summary.connections_by_vpn_status == {"nonvpn": 1, "vpn": 2}
    assert summary.connections_by_category == {"C2": 1, "STREAMING": 2}
    assert summary.connections_by_application == {"ssh": 1, "youtube": 2}


def test_inspect_raw_dataframe_rejects_wrong_columns() -> None:
    dataframe = _valid_dataframe().drop(columns="directions")

    with pytest.raises(VnatDatasetError, match="expected columns"):
        inspect_raw_dataframe(dataframe)


def test_inspect_raw_dataframe_rejects_empty_data() -> None:
    dataframe = pd.DataFrame(columns=RAW_COLUMNS)

    with pytest.raises(VnatDatasetError, match="at least one connection"):
        inspect_raw_dataframe(dataframe)


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        (["not", "a", "tuple"], "connection must be a tuple"),
        (("too", "short"), "exactly five values"),
    ],
)
def test_inspect_raw_dataframe_rejects_invalid_connections(
    connection: object, message: str
) -> None:
    dataframe = _valid_dataframe()
    cast(Any, dataframe).at[0, "connection"] = connection

    with pytest.raises(VnatDatasetError, match=message):
        inspect_raw_dataframe(dataframe)


@pytest.mark.parametrize("column", ["timestamps", "sizes", "directions"])
def test_inspect_raw_dataframe_requires_packet_lists(column: str) -> None:
    dataframe = _valid_dataframe()
    cast(Any, dataframe).at[0, column] = "not-a-list"

    with pytest.raises(VnatDatasetError, match=f"{column} must be a list"):
        inspect_raw_dataframe(dataframe)


def test_inspect_raw_dataframe_rejects_misaligned_packet_arrays() -> None:
    dataframe = _valid_dataframe()
    cast(Any, dataframe).at[0, "sizes"] = [100]

    with pytest.raises(VnatDatasetError, match="packet arrays must have equal lengths"):
        inspect_raw_dataframe(dataframe)


def test_inspect_raw_dataframe_rejects_empty_connections() -> None:
    dataframe = _valid_dataframe()
    for column in ("timestamps", "sizes", "directions"):
        cast(Any, dataframe).at[0, column] = []

    with pytest.raises(VnatDatasetError, match="connection must contain packets"):
        inspect_raw_dataframe(dataframe)


def test_inspect_raw_dataframe_requires_string_filenames() -> None:
    dataframe = _valid_dataframe()
    cast(Any, dataframe).at[0, "file_names"] = 123

    with pytest.raises(VnatDatasetError, match="file_names must contain strings"):
        inspect_raw_dataframe(dataframe)


def test_inspect_raw_dataframe_wraps_filename_errors() -> None:
    dataframe = _valid_dataframe()
    cast(Any, dataframe).at[0, "file_names"] = "unknown_capture.pcap"

    with pytest.raises(VnatDatasetError, match="expected a vpn_ or nonvpn_ prefix"):
        inspect_raw_dataframe(dataframe)


def test_inspect_vnat_file_reports_provenance(tmp_path: Path) -> None:
    path = tmp_path / "VNAT_Dataframe_release_1.h5"
    with catch_warnings():
        simplefilter("ignore", pd.errors.PerformanceWarning)
        _valid_dataframe().to_hdf(path, key="data")

    report = inspect_vnat_file(path)
    serialized = report.as_dict()

    assert report.source == str(path)
    assert report.file_size_bytes == path.stat().st_size
    assert len(report.sha256) == 64
    assert report.summary.connections == 3
    assert serialized["sha256"] == report.sha256


def test_inspect_vnat_file_requires_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.h5"

    with pytest.raises(VnatDatasetError, match="does not exist"):
        inspect_vnat_file(path)


def test_inspect_vnat_file_wraps_hdf_errors(tmp_path: Path) -> None:
    path = tmp_path / "wrong-key.h5"
    with catch_warnings():
        simplefilter("ignore", pd.errors.PerformanceWarning)
        _valid_dataframe().to_hdf(path, key="wrong_key")

    with pytest.raises(VnatDatasetError, match="could not read VNAT dataframe"):
        inspect_vnat_file(path)


def test_inspect_vnat_file_requires_dataframe_content(tmp_path: Path) -> None:
    path = tmp_path / "series.h5"
    pd.Series([1, 2, 3]).to_hdf(path, key="data")

    with pytest.raises(VnatDatasetError, match="expected a dataframe"):
        inspect_vnat_file(path)
