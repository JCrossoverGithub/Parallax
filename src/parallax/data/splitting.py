"""Deterministic capture-grouped partition assignment for VNAT artifacts."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, milp

from parallax.data.vnat import (
    APPLICATION_TO_CATEGORY,
    Application,
    TrafficCategory,
    VpnStatus,
)

CAPTURE_SPLIT_SCHEMA_VERSION: Final = "vnat-capture-split-1"


class DatasetPartition(StrEnum):
    """Independent roles in the primary model-development split."""

    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


DATASET_PARTITIONS: Final = tuple(DatasetPartition)


class CaptureSplitError(ValueError):
    """Raised when capture groups cannot produce a valid partition assignment."""


@dataclass(frozen=True, slots=True)
class CaptureGroup:
    """One indivisible source capture and its feature-window count."""

    capture_id: str
    vpn_status: VpnStatus
    application: Application
    category: TrafficCategory
    windows: int


@dataclass(frozen=True, slots=True)
class CaptureSplitConfig:
    """Versioned targets and hard coverage floor for the primary split."""

    train_fraction: float = 0.60
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.10
    test_fraction: float = 0.15
    minimum_category_windows: int = 20
    relative_mip_gap: float = 0.10
    solver_time_limit_seconds: float = 15.0

    def __post_init__(self) -> None:
        fractions = self.fractions
        if any(not 0.0 < fraction < 1.0 for fraction in fractions.values()):
            raise CaptureSplitError("partition fractions must be between zero and one")
        if not isclose(sum(fractions.values()), 1.0, abs_tol=1e-12):
            raise CaptureSplitError("partition fractions must sum to one")
        if self.minimum_category_windows < 1:
            raise CaptureSplitError("minimum category windows must be positive")
        if not 0.0 <= self.relative_mip_gap < 1.0:
            raise CaptureSplitError("relative MIP gap must be between zero and one")
        if self.solver_time_limit_seconds <= 0.0:
            raise CaptureSplitError("solver time limit must be positive")

    @property
    def fractions(self) -> dict[DatasetPartition, float]:
        """Return partition fractions in canonical order."""
        return {
            DatasetPartition.TRAIN: self.train_fraction,
            DatasetPartition.VALIDATION: self.validation_fraction,
            DatasetPartition.CALIBRATION: self.calibration_fraction,
            DatasetPartition.TEST: self.test_fraction,
        }


@dataclass(frozen=True, slots=True)
class CaptureAssignment:
    """The assigned partition for one source capture."""

    capture: CaptureGroup
    partition: DatasetPartition


@dataclass(frozen=True, slots=True)
class CaptureSplitResult:
    """A deterministic feasible solution returned by the split optimizer."""

    assignments: tuple[CaptureAssignment, ...]
    objective_value: float
    optimal: bool
    solver_message: str

    def partition_for(self, capture_id: str) -> DatasetPartition:
        """Return the partition assigned to a known capture identifier."""
        for assignment in self.assignments:
            if assignment.capture.capture_id == capture_id:
                return assignment.partition
        raise KeyError(capture_id)


@dataclass(slots=True)
class _ConstraintRows:
    coefficients: list[npt.NDArray[np.float64]]
    lower_bounds: list[float]
    upper_bounds: list[float]

    def add(
        self,
        coefficients: npt.NDArray[np.float64],
        *,
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        self.coefficients.append(coefficients)
        self.lower_bounds.append(lower)
        self.upper_bounds.append(upper)


@dataclass(frozen=True, slots=True)
class _BalanceMetric:
    coefficients: npt.NDArray[np.float64]
    target: float
    objective_weight: float


def assign_capture_splits(
    captures: Iterable[CaptureGroup],
    *,
    config: CaptureSplitConfig | None = None,
) -> CaptureSplitResult:
    """Assign whole captures with hard coverage and soft balance constraints."""
    selected_config = config or CaptureSplitConfig()
    ordered = _validate_captures(captures)
    partition_count = len(DATASET_PARTITIONS)
    binary_count = len(ordered) * partition_count
    metrics = _balance_metrics(ordered, selected_config, binary_count)
    variable_count = binary_count + len(metrics)
    rows = _ConstraintRows([], [], [])

    _add_assignment_constraints(rows, ordered, variable_count)
    _add_coverage_constraints(rows, ordered, selected_config, variable_count)
    _add_absolute_deviation_constraints(rows, metrics, binary_count, variable_count)

    objective = np.zeros(variable_count, dtype=np.float64)
    for metric_index, metric in enumerate(metrics):
        objective[binary_count + metric_index] = metric.objective_weight

    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    upper_bounds[:binary_count] = 1.0
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:binary_count] = 1
    constraint = LinearConstraint(
        np.vstack(rows.coefficients),
        lb=np.asarray(rows.lower_bounds),
        ub=np.asarray(rows.upper_bounds),
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraint,
        options={
            "presolve": True,
            "mip_rel_gap": selected_config.relative_mip_gap,
            "time_limit": selected_config.solver_time_limit_seconds,
        },
    )

    if result.x is None:
        raise CaptureSplitError(f"capture split optimization failed: {result.message}")

    assignments = _decode_assignments(ordered, result.x[:binary_count])
    validate_capture_assignments(assignments, config=selected_config)
    return CaptureSplitResult(
        assignments=assignments,
        objective_value=float(result.fun),
        optimal=bool(result.success),
        solver_message=result.message,
    )


def _validate_captures(captures: Iterable[CaptureGroup]) -> tuple[CaptureGroup, ...]:
    ordered = tuple(sorted(captures, key=lambda capture: capture.capture_id))
    if not ordered:
        raise CaptureSplitError("at least one capture group is required")

    identifiers = [capture.capture_id for capture in ordered]
    if any(not identifier for identifier in identifiers):
        raise CaptureSplitError("capture identifiers must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise CaptureSplitError("capture identifiers must be unique")

    for capture in ordered:
        if capture.windows < 1:
            raise CaptureSplitError("capture window counts must be positive")
        if APPLICATION_TO_CATEGORY[capture.application] is not capture.category:
            raise CaptureSplitError(
                f"application/category mismatch for capture {capture.capture_id!r}"
            )
    return ordered


def _binary_index(capture_index: int, partition_index: int) -> int:
    return capture_index * len(DATASET_PARTITIONS) + partition_index


def _coefficient_row(variable_count: int) -> npt.NDArray[np.float64]:
    return np.zeros(variable_count, dtype=np.float64)


def _matching_coefficients(
    captures: tuple[CaptureGroup, ...],
    variable_count: int,
    *,
    partition: DatasetPartition,
    category: TrafficCategory | None = None,
    application: Application | None = None,
    vpn_status: VpnStatus | None = None,
    use_windows: bool = False,
) -> npt.NDArray[np.float64]:
    coefficients = _coefficient_row(variable_count)
    partition_index = DATASET_PARTITIONS.index(partition)
    for capture_index, capture in enumerate(captures):
        if category is not None and capture.category is not category:
            continue
        if application is not None and capture.application is not application:
            continue
        if vpn_status is not None and capture.vpn_status is not vpn_status:
            continue
        coefficients[_binary_index(capture_index, partition_index)] = (
            float(capture.windows) if use_windows else 1.0
        )
    return coefficients


def _add_assignment_constraints(
    rows: _ConstraintRows,
    captures: tuple[CaptureGroup, ...],
    variable_count: int,
) -> None:
    for capture_index in range(len(captures)):
        coefficients = _coefficient_row(variable_count)
        for partition_index in range(len(DATASET_PARTITIONS)):
            coefficients[_binary_index(capture_index, partition_index)] = 1.0
        rows.add(coefficients, lower=1.0, upper=1.0)


def _add_coverage_constraints(
    rows: _ConstraintRows,
    captures: tuple[CaptureGroup, ...],
    config: CaptureSplitConfig,
    variable_count: int,
) -> None:
    categories = tuple(sorted(TrafficCategory, key=str))
    applications = tuple(sorted(Application, key=str))
    vpn_statuses = tuple(sorted(VpnStatus, key=str))

    for category in categories:
        for partition in DATASET_PARTITIONS:
            rows.add(
                _matching_coefficients(
                    captures,
                    variable_count,
                    partition=partition,
                    category=category,
                ),
                lower=1.0,
            )
            rows.add(
                _matching_coefficients(
                    captures,
                    variable_count,
                    partition=partition,
                    category=category,
                    use_windows=True,
                ),
                lower=float(config.minimum_category_windows),
            )

    for application in applications:
        for partition in (DatasetPartition.TRAIN, DatasetPartition.TEST):
            rows.add(
                _matching_coefficients(
                    captures,
                    variable_count,
                    partition=partition,
                    application=application,
                ),
                lower=1.0,
            )

    for partition in DATASET_PARTITIONS:
        for vpn_status in vpn_statuses:
            rows.add(
                _matching_coefficients(
                    captures,
                    variable_count,
                    partition=partition,
                    vpn_status=vpn_status,
                ),
                lower=1.0,
            )

    for category in categories:
        for vpn_status in vpn_statuses:
            rows.add(
                _matching_coefficients(
                    captures,
                    variable_count,
                    partition=DatasetPartition.TRAIN,
                    category=category,
                    vpn_status=vpn_status,
                ),
                lower=1.0,
            )


def _balance_metrics(
    captures: tuple[CaptureGroup, ...],
    config: CaptureSplitConfig,
    binary_count: int,
) -> tuple[_BalanceMetric, ...]:
    metrics: list[_BalanceMetric] = []
    fractions = config.fractions
    categories = tuple(sorted(TrafficCategory, key=str))
    vpn_statuses = tuple(sorted(VpnStatus, key=str))

    def add_metrics(
        *,
        matching: tuple[CaptureGroup, ...],
        category: TrafficCategory | None = None,
        vpn_status: VpnStatus | None = None,
        use_windows: bool,
        importance: float,
    ) -> None:
        total = sum(capture.windows if use_windows else 1 for capture in matching)
        if total == 0:
            return
        for partition in DATASET_PARTITIONS:
            metrics.append(
                _BalanceMetric(
                    coefficients=_matching_coefficients(
                        captures,
                        binary_count,
                        partition=partition,
                        category=category,
                        vpn_status=vpn_status,
                        use_windows=use_windows,
                    ),
                    target=total * fractions[partition],
                    objective_weight=importance / total,
                )
            )

    for category in categories:
        matching = tuple(capture for capture in captures if capture.category is category)
        add_metrics(matching=matching, category=category, use_windows=True, importance=10.0)
        add_metrics(matching=matching, category=category, use_windows=False, importance=2.0)

    add_metrics(matching=captures, use_windows=True, importance=5.0)
    add_metrics(matching=captures, use_windows=False, importance=1.0)

    for vpn_status in vpn_statuses:
        matching = tuple(capture for capture in captures if capture.vpn_status is vpn_status)
        add_metrics(
            matching=matching,
            vpn_status=vpn_status,
            use_windows=True,
            importance=1.0,
        )

    return tuple(metrics)


def _add_absolute_deviation_constraints(
    rows: _ConstraintRows,
    metrics: tuple[_BalanceMetric, ...],
    binary_count: int,
    variable_count: int,
) -> None:
    for metric_index, metric in enumerate(metrics):
        deviation_index = binary_count + metric_index
        positive = _coefficient_row(variable_count)
        positive[:binary_count] = metric.coefficients
        positive[deviation_index] = -1.0
        rows.add(positive, upper=metric.target)

        negative = _coefficient_row(variable_count)
        negative[:binary_count] = -metric.coefficients
        negative[deviation_index] = -1.0
        rows.add(negative, upper=-metric.target)


def _decode_assignments(
    captures: tuple[CaptureGroup, ...], binary_values: npt.NDArray[np.float64]
) -> tuple[CaptureAssignment, ...]:
    matrix = np.rint(binary_values).astype(np.int8).reshape(len(captures), len(DATASET_PARTITIONS))
    assignments: list[CaptureAssignment] = []
    for capture, row in zip(captures, matrix, strict=True):
        selected = np.flatnonzero(row)
        if len(selected) != 1:
            raise CaptureSplitError("solver returned an invalid capture assignment")
        assignments.append(
            CaptureAssignment(
                capture=capture,
                partition=DATASET_PARTITIONS[int(selected[0])],
            )
        )
    return tuple(assignments)


def validate_capture_assignments(
    assignments: Iterable[CaptureAssignment],
    *,
    config: CaptureSplitConfig | None = None,
) -> None:
    """Reject an assignment that violates the versioned split contract."""
    selected = tuple(assignments)
    selected_config = config or CaptureSplitConfig()
    for category in TrafficCategory:
        for partition in DATASET_PARTITIONS:
            matching = [
                assignment
                for assignment in selected
                if assignment.capture.category is category and assignment.partition is partition
            ]
            if not matching:
                raise CaptureSplitError("solver result does not cover every category partition")
            if (
                sum(item.capture.windows for item in matching)
                < selected_config.minimum_category_windows
            ):
                raise CaptureSplitError("solver result violates the category window floor")

    for partition in (DatasetPartition.TRAIN, DatasetPartition.TEST):
        observed_applications = {
            assignment.capture.application
            for assignment in selected
            if assignment.partition is partition
        }
        if observed_applications != set(Application):
            raise CaptureSplitError(
                "solver result does not cover every training and test application"
            )

    for partition in DATASET_PARTITIONS:
        observed = {
            assignment.capture.vpn_status
            for assignment in selected
            if assignment.partition is partition
        }
        if observed != set(VpnStatus):
            raise CaptureSplitError("solver result does not cover both VPN statuses")

    for category in TrafficCategory:
        observed = {
            assignment.capture.vpn_status
            for assignment in selected
            if assignment.partition is DatasetPartition.TRAIN
            and assignment.capture.category is category
        }
        if observed != set(VpnStatus):
            raise CaptureSplitError("training does not cover each category and VPN status")
