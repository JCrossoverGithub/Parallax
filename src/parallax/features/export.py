"""Versioned Parquet export for calculated VNAT feature vectors."""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from parallax.data.window_export import WINDOW_PARQUET_SCHEMA
from parallax.features.calculator import (
    FeatureCalculationConfig,
    calculate_feature_vector,
)
from parallax.features.schema import FEATURE_COLUMNS

FEATURE_ARTIFACT_SCHEMA_VERSION: Final = "vnat-feature-artifact-1"
RELEASE_COMPATIBLE_WINDOW_SHA256: Final = (
    "06f00af45cb635241575d251331e7ce96273212dba087e38b7610876ec9984d8"
)
FEATURE_PARQUET_SCHEMA: Final = pa.schema(
    cast(
        "list[pa.Field[Any]]",
        [
            pa.field("window_id", pa.string(), nullable=False),
            pa.field("capture_id", pa.string(), nullable=False),
            pa.field("flow_id", pa.string(), nullable=False),
            pa.field("window_index", pa.int64(), nullable=False),
            pa.field("vpn_status", pa.string(), nullable=False),
            pa.field("application", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("packet_count", pa.int64(), nullable=False),
            *(pa.field(column, pa.float32(), nullable=False) for column in FEATURE_COLUMNS),
        ],
    ),
    metadata={b"parallax.schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION.encode()},
)


class FeatureExportError(ValueError):
    """Raised when a feature artifact cannot be created safely."""


@dataclass(frozen=True, slots=True)
class FeatureExportSummary:
    """Auditable counts for one calculated feature artifact."""

    windows: int
    packets: int
    captures: int
    windows_by_vpn_status: dict[str, int]
    windows_by_application: dict[str, int]
    windows_by_category: dict[str, int]


@dataclass(frozen=True, slots=True)
class FeatureExportReport:
    """Provenance and output metadata written beside a feature artifact."""

    source: str
    source_file_size_bytes: int
    source_sha256: str
    output: str
    output_file_size_bytes: int
    output_sha256: str
    manifest: str
    config: FeatureCalculationConfig
    summary: FeatureExportSummary

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON manifest representation."""
        return {
            "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
            "source": {
                "path": self.source,
                "file_size_bytes": self.source_file_size_bytes,
                "sha256": self.source_sha256,
                "schema_version": _window_schema_version(),
            },
            "configuration": {
                "window_seconds": self.config.window_seconds,
                "time_bin_seconds": self.config.time_bin_seconds,
                "byte_total_policy": self.config.byte_total_policy.value,
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


def export_vnat_features(
    source: str | Path,
    output: str | Path,
    *,
    expected_sha256: str,
    config: FeatureCalculationConfig | None = None,
) -> FeatureExportReport:
    """Verify a window artifact and atomically export calculated features."""
    source_path = Path(source)
    destination = Path(output)
    manifest = destination.with_suffix(".manifest.json")
    selected_config = config or FeatureCalculationConfig()
    _validate_destinations(destination, manifest)

    source_sha256 = _sha256_file(source_path)
    if source_sha256 != expected_sha256:
        raise FeatureExportError(
            f"source SHA-256 mismatch: expected {expected_sha256}, got {source_sha256}"
        )

    parquet = pq.ParquetFile(source_path)
    if not _matches_window_schema(parquet.schema_arrow):
        raise FeatureExportError("source does not match the versioned window schema")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".parallax-feature-export-", dir=destination.parent
    ) as temporary:
        temporary_directory = Path(temporary)
        temporary_output = temporary_directory / destination.name
        temporary_manifest = temporary_directory / manifest.name
        summary = _write_features(parquet, temporary_output, selected_config)
        output_sha256 = _sha256_file(temporary_output)
        report = FeatureExportReport(
            source=str(source_path),
            source_file_size_bytes=source_path.stat().st_size,
            source_sha256=source_sha256,
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


def _write_features(
    parquet: pq.ParquetFile,
    output: Path,
    config: FeatureCalculationConfig,
) -> FeatureExportSummary:
    writer = pq.ParquetWriter(
        output,
        FEATURE_PARQUET_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    windows = 0
    packets = 0
    captures: set[str] = set()
    by_vpn_status: Counter[str] = Counter()
    by_application: Counter[str] = Counter()
    by_category: Counter[str] = Counter()

    try:
        for row_group in range(parquet.num_row_groups):
            records = parquet.read_row_group(row_group).to_pylist()
            writer.write_table(_feature_table(records, config))
            for record in records:
                windows += 1
                packets += int(record["packet_count"])
                captures.add(str(record["capture_id"]))
                by_vpn_status[str(record["vpn_status"])] += 1
                by_application[str(record["application"])] += 1
                by_category[str(record["category"])] += 1
    finally:
        writer.close()

    return FeatureExportSummary(
        windows=windows,
        packets=packets,
        captures=len(captures),
        windows_by_vpn_status=_sorted_counts(by_vpn_status),
        windows_by_application=_sorted_counts(by_application),
        windows_by_category=_sorted_counts(by_category),
    )


def _feature_table(
    records: list[dict[str, object]],
    config: FeatureCalculationConfig,
) -> pa.Table:
    columns: dict[str, list[object]] = {name: [] for name in FEATURE_PARQUET_SCHEMA.names}

    for record in records:
        timestamps = cast("list[float]", record["timestamps"])
        sizes = cast("list[int]", record["sizes"])
        directions = cast("list[int]", record["directions"])
        packet_count = cast("int", record["packet_count"])
        if not len(timestamps) == len(sizes) == len(directions) == packet_count:
            raise FeatureExportError("window packet count does not match packet arrays")

        vector = calculate_feature_vector(
            timestamps,
            sizes,
            directions,
            config=config,
        )
        for name in (
            "window_id",
            "capture_id",
            "flow_id",
            "window_index",
            "vpn_status",
            "application",
            "category",
            "packet_count",
        ):
            columns[name].append(record[name])
        for name, value in zip(FEATURE_COLUMNS, vector, strict=True):
            columns[name].append(float(value))

    return pa.Table.from_pydict(columns, schema=FEATURE_PARQUET_SCHEMA)


def _validate_destinations(output: Path, manifest: Path) -> None:
    if output.suffix.casefold() != ".parquet":
        raise FeatureExportError("feature output must use the .parquet extension")
    if output.exists():
        raise FeatureExportError(f"feature output already exists: {output}")
    if manifest.exists():
        raise FeatureExportError(f"feature manifest already exists: {manifest}")


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_schema_version() -> str:
    metadata = WINDOW_PARQUET_SCHEMA.metadata or {}
    return metadata[b"parallax.schema_version"].decode()


def _matches_window_schema(schema: pa.Schema) -> bool:
    metadata = schema.metadata or {}
    return (
        schema.equals(WINDOW_PARQUET_SCHEMA, check_metadata=False)
        and metadata.get(b"parallax.schema_version") == _window_schema_version().encode()
    )
