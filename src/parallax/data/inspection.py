"""Inspection and structural validation for VNAT raw dataframes."""

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import cast

import pandas as pd

from parallax.data.vnat import (
    RAW_COLUMNS,
    CaptureMetadata,
    VnatFilenameError,
    parse_capture_filename,
)


class VnatDatasetError(ValueError):
    """Raised when a VNAT dataframe violates the raw-data contract."""


@dataclass(frozen=True, slots=True)
class VnatDatasetSummary:
    """Content summary produced after validating a raw VNAT dataframe."""

    connections: int
    packets: int
    captures: int
    minimum_packets_per_connection: int
    median_packets_per_connection: float
    maximum_packets_per_connection: int
    captures_by_vpn_status: dict[str, int]
    captures_by_category: dict[str, int]
    captures_by_application: dict[str, int]
    connections_by_vpn_status: dict[str, int]
    connections_by_category: dict[str, int]
    connections_by_application: dict[str, int]


@dataclass(frozen=True, slots=True)
class VnatInspectionReport:
    """File provenance and validated content summary."""

    source: str
    file_size_bytes: int
    sha256: str
    summary: VnatDatasetSummary

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this report."""
        return asdict(self)


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _require_packet_list(value: object, *, column: str, row_number: int) -> list[object]:
    if not isinstance(value, list):
        raise VnatDatasetError(f"row {row_number}: {column} must be a list")
    return cast(list[object], value)


def inspect_raw_dataframe(dataframe: pd.DataFrame) -> VnatDatasetSummary:
    """Validate one raw VNAT dataframe and summarize its contents."""
    columns = tuple(str(column) for column in dataframe.columns)
    if columns != RAW_COLUMNS:
        raise VnatDatasetError(f"expected columns {RAW_COLUMNS!r}, got {columns!r}")
    if dataframe.empty:
        raise VnatDatasetError("raw VNAT dataframe must contain at least one connection")

    packet_counts: list[int] = []
    capture_metadata: dict[str, CaptureMetadata] = {}
    connection_vpn_counts: Counter[str] = Counter()
    connection_category_counts: Counter[str] = Counter()
    connection_application_counts: Counter[str] = Counter()

    for row_number, row in enumerate(dataframe.itertuples(index=False, name=None), start=1):
        connection, raw_timestamps, raw_sizes, raw_directions, raw_file_name = row

        if not isinstance(connection, tuple):
            raise VnatDatasetError(f"row {row_number}: connection must be a tuple")
        if len(connection) != 5:
            raise VnatDatasetError(f"row {row_number}: connection must contain exactly five values")

        timestamps = _require_packet_list(
            raw_timestamps, column="timestamps", row_number=row_number
        )
        sizes = _require_packet_list(raw_sizes, column="sizes", row_number=row_number)
        directions = _require_packet_list(
            raw_directions, column="directions", row_number=row_number
        )

        lengths = {len(timestamps), len(sizes), len(directions)}
        if len(lengths) != 1:
            raise VnatDatasetError(f"row {row_number}: packet arrays must have equal lengths")

        packet_count = len(timestamps)
        if packet_count == 0:
            raise VnatDatasetError(f"row {row_number}: connection must contain packets")

        if not isinstance(raw_file_name, str):
            raise VnatDatasetError(f"row {row_number}: file_names must contain strings")
        try:
            metadata = parse_capture_filename(raw_file_name)
        except VnatFilenameError as error:
            raise VnatDatasetError(f"row {row_number}: {error}") from error

        packet_counts.append(packet_count)
        capture_metadata[metadata.capture_id] = metadata
        connection_vpn_counts[metadata.vpn_status.value] += 1
        connection_category_counts[metadata.category.value] += 1
        connection_application_counts[metadata.application.value] += 1

    capture_vpn_counts = Counter(item.vpn_status.value for item in capture_metadata.values())
    capture_category_counts = Counter(item.category.value for item in capture_metadata.values())
    capture_application_counts = Counter(
        item.application.value for item in capture_metadata.values()
    )

    return VnatDatasetSummary(
        connections=len(dataframe),
        packets=sum(packet_counts),
        captures=len(capture_metadata),
        minimum_packets_per_connection=min(packet_counts),
        median_packets_per_connection=float(median(packet_counts)),
        maximum_packets_per_connection=max(packet_counts),
        captures_by_vpn_status=_sorted_counts(capture_vpn_counts),
        captures_by_category=_sorted_counts(capture_category_counts),
        captures_by_application=_sorted_counts(capture_application_counts),
        connections_by_vpn_status=_sorted_counts(connection_vpn_counts),
        connections_by_category=_sorted_counts(connection_category_counts),
        connections_by_application=_sorted_counts(connection_application_counts),
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_vnat_file(path: str | Path, *, expected_sha256: str) -> VnatInspectionReport:
    """Load, validate, and summarize a raw VNAT HDF5 file."""
    source = Path(path)
    if not source.is_file():
        raise VnatDatasetError(f"VNAT dataset file does not exist: {source}")

    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256.casefold():
        raise VnatDatasetError(
            f"SHA-256 mismatch for {source}: expected {expected_sha256.casefold()}, "
            f"got {actual_sha256}"
        )

    try:
        loaded = pd.read_hdf(source, key="data")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise VnatDatasetError(f"could not read VNAT dataframe from {source}") from error
    if not isinstance(loaded, pd.DataFrame):
        raise VnatDatasetError(f"expected a dataframe in {source}")

    return VnatInspectionReport(
        source=str(source),
        file_size_bytes=source.stat().st_size,
        sha256=actual_sha256,
        summary=inspect_raw_dataframe(loaded),
    )
