"""Immutable manifests for deterministic capture-grouped feature splits."""

import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from parallax.data.splitting import (
    CAPTURE_SPLIT_SCHEMA_VERSION,
    DATASET_PARTITIONS,
    CaptureAssignment,
    CaptureGroup,
    CaptureSplitConfig,
    CaptureSplitError,
    CaptureSplitResult,
    assign_capture_splits,
)
from parallax.data.vnat import Application, TrafficCategory, VpnStatus
from parallax.features.export import (
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FEATURE_PARQUET_SCHEMA,
)

CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION: Final = "vnat-capture-split-manifest-1"
RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SHA256: Final = (
    "611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16"
)
RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SIZE_BYTES: Final = 14_702_124
_LABEL_COLUMNS: Final = (
    "capture_id",
    "vpn_status",
    "application",
    "category",
)


class CaptureSplitManifestError(ValueError):
    """Raised when a split manifest cannot be published safely."""


@dataclass(frozen=True, slots=True)
class CaptureSplitManifestReport:
    """Published split-manifest metadata and its audited assignment result."""

    source: str
    source_file_size_bytes: int
    source_sha256: str
    source_rows: int
    source_columns: int
    source_row_groups: int
    output: str
    output_file_size_bytes: int
    output_sha256: str
    config: CaptureSplitConfig
    result: CaptureSplitResult

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation written to the manifest."""
        return _manifest_payload(
            source_name=Path(self.source).name,
            source_file_size_bytes=self.source_file_size_bytes,
            source_sha256=self.source_sha256,
            source_rows=self.source_rows,
            source_columns=self.source_columns,
            source_row_groups=self.source_row_groups,
            config=self.config,
            result=self.result,
        )


@dataclass(slots=True)
class _CaptureAccumulator:
    vpn_status: VpnStatus
    application: Application
    category: TrafficCategory
    windows: int = 0


def export_capture_split_manifest(
    source: str | Path,
    output: str | Path,
    *,
    expected_sha256: str,
    config: CaptureSplitConfig | None = None,
) -> CaptureSplitManifestReport:
    """Verify a feature artifact and atomically publish its optimal split."""
    source_path = Path(source)
    destination = Path(output)
    selected_config = config or CaptureSplitConfig()
    _validate_destination(destination)

    source_sha256 = _sha256_file(source_path)
    if source_sha256 != expected_sha256:
        raise CaptureSplitManifestError(
            f"source SHA-256 mismatch: expected {expected_sha256}, got {source_sha256}"
        )

    parquet = pq.ParquetFile(source_path)
    if not _matches_feature_schema(parquet.schema_arrow):
        raise CaptureSplitManifestError("source does not match the versioned feature schema")

    captures = _read_capture_groups(parquet)
    try:
        result = assign_capture_splits(captures, config=selected_config)
    except CaptureSplitError as error:
        raise CaptureSplitManifestError(f"capture split failed: {error}") from error
    if not result.optimal:
        raise CaptureSplitManifestError(
            f"refusing to publish a split without proven optimality: {result.solver_message}"
        )

    metadata = parquet.metadata
    payload = _manifest_payload(
        source_name=source_path.name,
        source_file_size_bytes=source_path.stat().st_size,
        source_sha256=source_sha256,
        source_rows=metadata.num_rows,
        source_columns=metadata.num_columns,
        source_row_groups=metadata.num_row_groups,
        config=selected_config,
        result=result,
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".parallax-split-manifest-", dir=destination.parent
    ) as temporary:
        temporary_output = Path(temporary) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(temporary_output, destination)

    return CaptureSplitManifestReport(
        source=str(source_path),
        source_file_size_bytes=source_path.stat().st_size,
        source_sha256=source_sha256,
        source_rows=metadata.num_rows,
        source_columns=metadata.num_columns,
        source_row_groups=metadata.num_row_groups,
        output=str(destination),
        output_file_size_bytes=len(encoded),
        output_sha256=sha256(encoded).hexdigest(),
        config=selected_config,
        result=result,
    )


def _validate_destination(destination: Path) -> None:
    if destination.suffix.casefold() != ".json":
        raise CaptureSplitManifestError("split manifest output must use the .json extension")
    if destination.exists():
        raise CaptureSplitManifestError(f"split manifest already exists: {destination}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_feature_schema(schema: pa.Schema) -> bool:
    metadata = schema.metadata or {}
    return (
        schema.equals(FEATURE_PARQUET_SCHEMA, check_metadata=False)
        and metadata.get(b"parallax.schema_version") == FEATURE_ARTIFACT_SCHEMA_VERSION.encode()
    )


def _read_capture_groups(parquet: pq.ParquetFile) -> tuple[CaptureGroup, ...]:
    grouped: dict[str, _CaptureAccumulator] = {}
    row_count = 0

    for batch in parquet.iter_batches(columns=list(_LABEL_COLUMNS), batch_size=4096):
        for record in batch.to_pylist():
            row_count += 1
            capture_id = str(record["capture_id"])
            try:
                labels = _CaptureAccumulator(
                    vpn_status=VpnStatus(str(record["vpn_status"])),
                    application=Application(str(record["application"])),
                    category=TrafficCategory(str(record["category"])),
                )
            except ValueError as error:
                raise CaptureSplitManifestError(
                    f"invalid label for capture {capture_id!r}: {error}"
                ) from error

            current = grouped.setdefault(capture_id, labels)
            if (
                current.vpn_status is not labels.vpn_status
                or current.application is not labels.application
                or current.category is not labels.category
            ):
                raise CaptureSplitManifestError(f"inconsistent labels for capture {capture_id!r}")
            current.windows += 1

    if row_count == 0:
        raise CaptureSplitManifestError("feature artifact must contain at least one window")
    if row_count != parquet.metadata.num_rows:
        raise CaptureSplitManifestError("feature scan row count does not match Parquet metadata")

    return tuple(
        CaptureGroup(
            capture_id=capture_id,
            vpn_status=labels.vpn_status,
            application=labels.application,
            category=labels.category,
            windows=labels.windows,
        )
        for capture_id, labels in sorted(grouped.items())
    )


def _manifest_payload(
    *,
    source_name: str,
    source_file_size_bytes: int,
    source_sha256: str,
    source_rows: int,
    source_columns: int,
    source_row_groups: int,
    config: CaptureSplitConfig,
    result: CaptureSplitResult,
) -> dict[str, object]:
    return {
        "schema_version": CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_schema_version": CAPTURE_SPLIT_SCHEMA_VERSION,
        "source": {
            "file_name": source_name,
            "format": "parquet",
            "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
            "file_size_bytes": source_file_size_bytes,
            "sha256": source_sha256,
            "rows": source_rows,
            "columns": source_columns,
            "row_groups": source_row_groups,
        },
        "configuration": {
            "fractions": {
                partition.value: config.fractions[partition] for partition in DATASET_PARTITIONS
            },
            "minimum_category_windows": config.minimum_category_windows,
            "relative_mip_gap": config.relative_mip_gap,
            "solver_time_limit_seconds": config.solver_time_limit_seconds,
        },
        "solver": {
            "objective_value": result.objective_value,
            "optimality_proven": result.optimal,
            "message": result.solver_message,
        },
        "summary": _split_summary(result.assignments),
        "assignments": [
            {
                "capture_id": assignment.capture.capture_id,
                "partition": assignment.partition.value,
                "vpn_status": assignment.capture.vpn_status.value,
                "application": assignment.capture.application.value,
                "category": assignment.capture.category.value,
                "windows": assignment.capture.windows,
            }
            for assignment in result.assignments
        ],
    }


def _split_summary(assignments: Iterable[CaptureAssignment]) -> dict[str, object]:
    selected = tuple(assignments)
    total_windows = sum(item.capture.windows for item in selected)
    partitions: dict[str, object] = {}

    for partition in DATASET_PARTITIONS:
        matching = tuple(item for item in selected if item.partition is partition)
        partition_windows = sum(item.capture.windows for item in matching)
        partitions[partition.value] = {
            "captures": len(matching),
            "windows": partition_windows,
            "window_fraction": partition_windows / total_windows,
            "captures_by_category": _counts(item.capture.category.value for item in matching),
            "windows_by_category": _weighted_counts(
                (item.capture.category.value, item.capture.windows) for item in matching
            ),
            "captures_by_vpn_status": _counts(item.capture.vpn_status.value for item in matching),
            "windows_by_vpn_status": _weighted_counts(
                (item.capture.vpn_status.value, item.capture.windows) for item in matching
            ),
            "captures_by_application": _counts(item.capture.application.value for item in matching),
            "windows_by_application": _weighted_counts(
                (item.capture.application.value, item.capture.windows) for item in matching
            ),
        }

    return {
        "captures": len(selected),
        "windows": total_windows,
        "partitions": partitions,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _weighted_counts(values: Iterable[tuple[str, int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for label, weight in values:
        counts[label] += weight
    return dict(sorted(counts.items()))
