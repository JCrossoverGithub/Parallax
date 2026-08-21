"""Verified, manifest-bound feature arrays for offline model development."""

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from parallax.data import (
    APPLICATION_TO_CATEGORY,
    CAPTURE_SPLIT_SCHEMA_VERSION,
    DATASET_PARTITIONS,
    Application,
    DatasetPartition,
    TrafficCategory,
    VpnStatus,
)
from parallax.features import (
    CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    FEATURE_PARQUET_SCHEMA,
)

MODELING_DATASET_SCHEMA_VERSION: Final = "vnat-modeling-dataset-1"
_IDENTITY_COLUMNS: Final = (
    "capture_id",
    "vpn_status",
    "application",
    "category",
)
_SCAN_COLUMNS: Final = _IDENTITY_COLUMNS + FEATURE_COLUMNS

Float32Matrix = npt.NDArray[np.float32]
StringVector = npt.NDArray[np.str_]


class ModelingDatasetError(ValueError):
    """Raised when features and their trusted split manifest do not agree."""


@dataclass(frozen=True, slots=True)
class FeaturePartition:
    """Read-only model inputs and provenance labels for one partition."""

    partition: DatasetPartition
    features: Float32Matrix
    categories: StringVector
    capture_ids: StringVector
    vpn_statuses: StringVector
    applications: StringVector

    @property
    def windows(self) -> int:
        """Return the number of feature rows in the partition."""
        return int(self.features.shape[0])

    @property
    def captures(self) -> int:
        """Return the number of distinct source captures in the partition."""
        return len(set(self.capture_ids.tolist()))


@dataclass(frozen=True, slots=True)
class PartitionedFeatureDataset:
    """A verified feature artifact separated by its immutable manifest."""

    feature_artifact_sha256: str
    split_manifest_sha256: str
    partitions: tuple[FeaturePartition, ...]

    def partition(self, partition: DatasetPartition) -> FeaturePartition:
        """Return one partition without combining it with another role."""
        for selected in self.partitions:
            if selected.partition is partition:
                return selected
        raise KeyError(partition)


@dataclass(frozen=True, slots=True)
class _ManifestSource:
    file_size_bytes: int
    sha256: str
    rows: int
    columns: int
    row_groups: int


@dataclass(frozen=True, slots=True)
class _ManifestAssignment:
    capture_id: str
    partition: DatasetPartition
    vpn_status: VpnStatus
    application: Application
    category: TrafficCategory
    windows: int


@dataclass(frozen=True, slots=True)
class _VerifiedManifest:
    source: _ManifestSource
    assignments: tuple[_ManifestAssignment, ...]


@dataclass(slots=True)
class _PartitionBuffers:
    features: list[Float32Matrix]
    categories: list[StringVector]
    capture_ids: list[StringVector]
    vpn_statuses: list[StringVector]
    applications: list[StringVector]


def load_partitioned_feature_dataset(
    feature_artifact: str | Path,
    split_manifest: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PartitionedFeatureDataset:
    """Verify provenance and return isolated, read-only modeling partitions."""
    feature_path = Path(feature_artifact)
    manifest_path = Path(split_manifest)

    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ModelingDatasetError(
            "split manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    manifest = _read_manifest(manifest_path)

    if feature_path.stat().st_size != manifest.source.file_size_bytes:
        raise ModelingDatasetError("feature artifact size does not match split manifest")
    feature_sha256 = _sha256_file(feature_path)
    if feature_sha256 != manifest.source.sha256:
        raise ModelingDatasetError("feature artifact SHA-256 does not match split manifest")

    parquet = pq.ParquetFile(feature_path)
    if not _matches_feature_schema(parquet.schema_arrow):
        raise ModelingDatasetError("feature artifact does not match the versioned feature schema")
    _validate_parquet_metadata(parquet, manifest.source)
    partitions = _scan_partitions(parquet, manifest.assignments)

    return PartitionedFeatureDataset(
        feature_artifact_sha256=feature_sha256,
        split_manifest_sha256=manifest_sha256,
        partitions=partitions,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> _VerifiedManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _parse_manifest(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ModelingDatasetError):
            raise
        raise ModelingDatasetError(f"invalid split manifest: {error}") from error


def _parse_manifest(payload: object) -> _VerifiedManifest:
    if not isinstance(payload, dict):
        raise ModelingDatasetError("split manifest root must be an object")
    if payload["schema_version"] != CAPTURE_SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ModelingDatasetError("unsupported split manifest schema version")
    if payload["split_schema_version"] != CAPTURE_SPLIT_SCHEMA_VERSION:
        raise ModelingDatasetError("unsupported capture split schema version")

    source_payload = cast("dict[str, object]", payload["source"])
    if source_payload["format"] != "parquet":
        raise ModelingDatasetError("split manifest source format must be parquet")
    if source_payload["schema_version"] != FEATURE_ARTIFACT_SCHEMA_VERSION:
        raise ModelingDatasetError("split manifest references an unsupported feature schema")
    source = _ManifestSource(
        file_size_bytes=int(cast("int", source_payload["file_size_bytes"])),
        sha256=str(source_payload["sha256"]),
        rows=int(cast("int", source_payload["rows"])),
        columns=int(cast("int", source_payload["columns"])),
        row_groups=int(cast("int", source_payload["row_groups"])),
    )
    if source.file_size_bytes < 1 or source.rows < 1 or source.columns < 1 or source.row_groups < 1:
        raise ModelingDatasetError("split manifest source counts must be positive")

    solver = cast("dict[str, object]", payload["solver"])
    if solver["optimality_proven"] is not True:
        raise ModelingDatasetError("split manifest does not contain a proven optimal solution")

    assignment_payloads = cast("list[dict[str, object]]", payload["assignments"])
    assignments = tuple(_parse_assignment(item) for item in assignment_payloads)
    if not assignments:
        raise ModelingDatasetError("split manifest must contain capture assignments")
    identifiers = [assignment.capture_id for assignment in assignments]
    if len(set(identifiers)) != len(identifiers):
        raise ModelingDatasetError("split manifest capture assignments must be unique")

    _validate_manifest_summary(payload, assignments, source.rows)
    return _VerifiedManifest(source=source, assignments=assignments)


def _parse_assignment(payload: dict[str, object]) -> _ManifestAssignment:
    assignment = _ManifestAssignment(
        capture_id=str(payload["capture_id"]),
        partition=DatasetPartition(str(payload["partition"])),
        vpn_status=VpnStatus(str(payload["vpn_status"])),
        application=Application(str(payload["application"])),
        category=TrafficCategory(str(payload["category"])),
        windows=int(cast("int", payload["windows"])),
    )
    if not assignment.capture_id:
        raise ModelingDatasetError("split manifest capture identifiers must not be empty")
    if assignment.windows < 1:
        raise ModelingDatasetError("split manifest capture window counts must be positive")
    if APPLICATION_TO_CATEGORY[assignment.application] is not assignment.category:
        raise ModelingDatasetError("split manifest application/category labels do not agree")
    return assignment


def _validate_manifest_summary(
    payload: dict[str, object],
    assignments: tuple[_ManifestAssignment, ...],
    source_rows: int,
) -> None:
    summary = cast("dict[str, object]", payload["summary"])
    total_windows = sum(assignment.windows for assignment in assignments)
    if int(cast("int", summary["captures"])) != len(assignments):
        raise ModelingDatasetError("split manifest capture summary does not match assignments")
    if int(cast("int", summary["windows"])) != total_windows or total_windows != source_rows:
        raise ModelingDatasetError("split manifest window summary does not match assignments")

    partition_summaries = cast("dict[str, dict[str, object]]", summary["partitions"])
    for partition in DATASET_PARTITIONS:
        matching = tuple(item for item in assignments if item.partition is partition)
        if not matching:
            raise ModelingDatasetError("every modeling partition must contain assignments")
        partition_summary = partition_summaries[partition.value]
        if int(cast("int", partition_summary["captures"])) != len(matching):
            raise ModelingDatasetError("split manifest partition capture summary does not match")
        if int(cast("int", partition_summary["windows"])) != sum(item.windows for item in matching):
            raise ModelingDatasetError("split manifest partition window summary does not match")


def _matches_feature_schema(schema: pa.Schema) -> bool:
    metadata = schema.metadata or {}
    return (
        schema.equals(FEATURE_PARQUET_SCHEMA, check_metadata=False)
        and metadata.get(b"parallax.schema_version") == FEATURE_ARTIFACT_SCHEMA_VERSION.encode()
    )


def _validate_parquet_metadata(parquet: pq.ParquetFile, source: _ManifestSource) -> None:
    metadata = parquet.metadata
    if (
        metadata.num_rows != source.rows
        or metadata.num_columns != source.columns
        or metadata.num_row_groups != source.row_groups
    ):
        raise ModelingDatasetError("feature artifact metadata does not match split manifest")


def _scan_partitions(
    parquet: pq.ParquetFile,
    assignments: tuple[_ManifestAssignment, ...],
) -> tuple[FeaturePartition, ...]:
    by_capture = {assignment.capture_id: assignment for assignment in assignments}
    observed: Counter[str] = Counter()
    buffers = {partition: _PartitionBuffers([], [], [], [], []) for partition in DATASET_PARTITIONS}

    for batch in parquet.iter_batches(columns=list(_SCAN_COLUMNS), batch_size=4096):
        records = batch.select(list(_IDENTITY_COLUMNS)).to_pylist()
        matrix = np.column_stack(
            [
                batch.column(batch.schema.get_field_index(column)).to_numpy(zero_copy_only=False)
                for column in FEATURE_COLUMNS
            ]
        ).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ModelingDatasetError("feature artifact contains non-finite feature values")

        partition_values: list[DatasetPartition] = []
        for record in records:
            capture_id = str(record["capture_id"])
            assignment = by_capture.get(capture_id)
            if assignment is None:
                raise ModelingDatasetError(
                    f"feature artifact contains unassigned capture {capture_id!r}"
                )
            if (
                str(record["vpn_status"]) != assignment.vpn_status.value
                or str(record["application"]) != assignment.application.value
                or str(record["category"]) != assignment.category.value
            ):
                raise ModelingDatasetError(
                    f"feature labels do not match manifest for capture {capture_id!r}"
                )
            observed[capture_id] += 1
            partition_values.append(assignment.partition)

        for partition in DATASET_PARTITIONS:
            indices = np.asarray(
                [index for index, selected in enumerate(partition_values) if selected is partition],
                dtype=np.intp,
            )
            if indices.size == 0:
                continue
            selected_records = [records[int(index)] for index in indices]
            selected = buffers[partition]
            selected.features.append(matrix[indices])
            selected.capture_ids.append(_string_vector(selected_records, "capture_id"))
            selected.vpn_statuses.append(_string_vector(selected_records, "vpn_status"))
            selected.applications.append(_string_vector(selected_records, "application"))
            selected.categories.append(_string_vector(selected_records, "category"))

    expected = Counter({assignment.capture_id: assignment.windows for assignment in assignments})
    if observed != expected:
        raise ModelingDatasetError("feature window counts do not match split manifest")

    return tuple(
        _freeze_partition(partition, buffers[partition]) for partition in DATASET_PARTITIONS
    )


def _string_vector(records: list[dict[str, object]], column: str) -> StringVector:
    return np.asarray([str(record[column]) for record in records], dtype=np.str_)


def _freeze_partition(partition: DatasetPartition, buffers: _PartitionBuffers) -> FeaturePartition:
    features = np.concatenate(buffers.features, axis=0)
    categories = np.concatenate(buffers.categories)
    capture_ids = np.concatenate(buffers.capture_ids)
    vpn_statuses = np.concatenate(buffers.vpn_statuses)
    applications = np.concatenate(buffers.applications)
    for array in (features, categories, capture_ids, vpn_statuses, applications):
        array.setflags(write=False)
    return FeaturePartition(
        partition=partition,
        features=features,
        categories=categories,
        capture_ids=capture_ids,
        vpn_statuses=vpn_statuses,
        applications=applications,
    )
