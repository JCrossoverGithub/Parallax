"""Versioned Parquet export for deterministic VNAT observation windows."""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parallax.data.inspection import (
    inspect_raw_dataframe,
    load_verified_vnat_dataframe,
)
from parallax.data.windowing import (
    ObservationWindow,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    extract_capture_windows,
)

WINDOW_SCHEMA_VERSION: Final = "vnat-window-1"
WINDOW_EXPORT_BATCH_SIZE: Final = 64
WINDOW_PARQUET_SCHEMA: Final = pa.schema(
    cast(
        "list[pa.Field[Any]]",
        [
            pa.field("window_id", pa.string(), nullable=False),
            pa.field("capture_id", pa.string(), nullable=False),
            pa.field("flow_id", pa.string(), nullable=False),
            pa.field("window_index", pa.int64(), nullable=False),
            pa.field("start_offset_seconds", pa.float64(), nullable=False),
            pa.field("end_offset_seconds", pa.float64(), nullable=False),
            pa.field("vpn_status", pa.string(), nullable=False),
            pa.field("application", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("packet_count", pa.int64(), nullable=False),
            pa.field("timestamps", pa.list_(pa.float64()), nullable=False),
            pa.field("sizes", pa.list_(pa.int64()), nullable=False),
            pa.field("directions", pa.list_(pa.int8()), nullable=False),
        ],
    ),
    metadata={b"parallax.schema_version": WINDOW_SCHEMA_VERSION.encode()},
)


class WindowExportError(ValueError):
    """Raised when a processed window artifact cannot be created safely."""


@dataclass(frozen=True, slots=True)
class WindowExportSummary:
    """Auditable counts for one processed window artifact."""

    source_connections: int
    source_packets: int
    source_captures: int
    represented_captures: int
    omitted_captures: list[str]
    windows: int
    packets: int
    windows_by_vpn_status: dict[str, int]
    windows_by_application: dict[str, int]
    windows_by_category: dict[str, int]


@dataclass(frozen=True, slots=True)
class WindowExportReport:
    """Provenance and output metadata written beside the Parquet artifact."""

    source: str
    source_file_size_bytes: int
    source_sha256: str
    output: str
    output_file_size_bytes: int
    output_sha256: str
    manifest: str
    config: WindowExtractionConfig
    summary: WindowExportSummary

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON manifest representation."""
        operator = (
            ">"
            if self.config.threshold_policy is WindowThresholdPolicy.RELEASE_COMPATIBLE
            else ">="
        )
        return {
            "schema_version": WINDOW_SCHEMA_VERSION,
            "source": {
                "path": self.source,
                "file_size_bytes": self.source_file_size_bytes,
                "sha256": self.source_sha256,
            },
            "configuration": {
                "window_seconds": self.config.window_seconds,
                "minimum_packets": self.config.minimum_packets,
                "eligibility_operator": operator,
                "threshold_policy": self.config.threshold_policy.value,
            },
            "output": {
                "path": self.output,
                "format": "parquet",
                "compression": "zstd",
                "file_size_bytes": self.output_file_size_bytes,
                "sha256": self.output_sha256,
            },
            "manifest": self.manifest,
            "summary": asdict(self.summary),
        }


def export_vnat_windows(
    source: str | Path,
    output: str | Path,
    *,
    expected_sha256: str,
    config: WindowExtractionConfig | None = None,
) -> WindowExportReport:
    """Verify a raw release and atomically export deterministic Parquet windows."""
    destination = Path(output)
    manifest = destination.with_suffix(".manifest.json")
    selected_config = config or WindowExtractionConfig()
    _validate_destinations(destination, manifest)

    verified = load_verified_vnat_dataframe(source, expected_sha256=expected_sha256)
    raw_summary = inspect_raw_dataframe(verified.dataframe)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".parallax-window-export-", dir=destination.parent) as temporary:
        temporary_directory = Path(temporary)
        temporary_output = temporary_directory / destination.name
        temporary_manifest = temporary_directory / manifest.name
        counters = _write_parquet(verified.dataframe, temporary_output, selected_config)
        output_sha256 = _sha256_file(temporary_output)

        omitted_captures = sorted(
            capture_id
            for capture_id, window_count in counters.capture_windows.items()
            if window_count == 0
        )
        summary = WindowExportSummary(
            source_connections=raw_summary.connections,
            source_packets=raw_summary.packets,
            source_captures=raw_summary.captures,
            represented_captures=raw_summary.captures - len(omitted_captures),
            omitted_captures=omitted_captures,
            windows=counters.windows,
            packets=counters.packets,
            windows_by_vpn_status=_sorted_counts(counters.by_vpn_status),
            windows_by_application=_sorted_counts(counters.by_application),
            windows_by_category=_sorted_counts(counters.by_category),
        )
        report = WindowExportReport(
            source=str(verified.source),
            source_file_size_bytes=verified.file_size_bytes,
            source_sha256=verified.sha256,
            output=str(destination),
            output_file_size_bytes=temporary_output.stat().st_size,
            output_sha256=output_sha256,
            manifest=str(manifest),
            config=selected_config,
            summary=summary,
        )
        temporary_manifest.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, destination)
        os.replace(temporary_manifest, manifest)

    return report


@dataclass(slots=True)
class _ExportCounters:
    windows: int
    packets: int
    capture_windows: Counter[str]
    by_vpn_status: Counter[str]
    by_application: Counter[str]
    by_category: Counter[str]


def _write_parquet(
    dataframe: pd.DataFrame,
    output: Path,
    config: WindowExtractionConfig,
) -> _ExportCounters:
    capture_ids = sorted(str(value) for value in dataframe["file_names"].unique())
    counters = _ExportCounters(
        windows=0,
        packets=0,
        capture_windows=Counter({capture_id: 0 for capture_id in capture_ids}),
        by_vpn_status=Counter(),
        by_application=Counter(),
        by_category=Counter(),
    )
    writer = pq.ParquetWriter(
        output,
        WINDOW_PARQUET_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    batch: list[ObservationWindow] = []

    try:
        for _, capture in dataframe.groupby("file_names", sort=True):
            for window in extract_capture_windows(capture, config=config):
                _record_window(counters, window)
                batch.append(window)
                if len(batch) == WINDOW_EXPORT_BATCH_SIZE:
                    writer.write_table(_window_table(batch))
                    batch.clear()

        if batch:
            writer.write_table(_window_table(batch))
    finally:
        writer.close()

    return counters


def _record_window(counters: _ExportCounters, window: ObservationWindow) -> None:
    counters.windows += 1
    counters.packets += window.packet_count
    counters.capture_windows[window.capture.capture_id] += 1
    counters.by_vpn_status[window.capture.vpn_status.value] += 1
    counters.by_application[window.capture.application.value] += 1
    counters.by_category[window.capture.category.value] += 1


def _window_table(windows: list[ObservationWindow]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "window_id": [window.window_id for window in windows],
            "capture_id": [window.capture.capture_id for window in windows],
            "flow_id": [window.flow_id for window in windows],
            "window_index": [window.window_index for window in windows],
            "start_offset_seconds": [window.start_offset_seconds for window in windows],
            "end_offset_seconds": [window.end_offset_seconds for window in windows],
            "vpn_status": [window.capture.vpn_status.value for window in windows],
            "application": [window.capture.application.value for window in windows],
            "category": [window.capture.category.value for window in windows],
            "packet_count": [window.packet_count for window in windows],
            "timestamps": [window.timestamps for window in windows],
            "sizes": [window.sizes for window in windows],
            "directions": [window.directions for window in windows],
        },
        schema=WINDOW_PARQUET_SCHEMA,
    )


def _validate_destinations(output: Path, manifest: Path) -> None:
    if output.suffix.casefold() != ".parquet":
        raise WindowExportError("window output must use the .parquet extension")
    if output.exists():
        raise WindowExportError(f"window output already exists: {output}")
    if manifest.exists():
        raise WindowExportError(f"window manifest already exists: {manifest}")


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
