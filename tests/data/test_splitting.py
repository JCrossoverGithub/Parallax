from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import parallax.data.splitting as splitting
from parallax.data import (
    APPLICATION_TO_CATEGORY,
    DATASET_PARTITIONS,
    Application,
    CaptureGroup,
    CaptureSplitConfig,
    CaptureSplitError,
    DatasetPartition,
    TrafficCategory,
    VpnStatus,
    assign_capture_splits,
    validate_capture_assignments,
)


def feasible_captures() -> list[CaptureGroup]:
    captures: list[CaptureGroup] = []
    for category in TrafficCategory:
        applications = [
            application
            for application, mapped_category in APPLICATION_TO_CATEGORY.items()
            if mapped_category is category
        ]
        for index in range(8):
            application = applications[index % len(applications)]
            captures.append(
                CaptureGroup(
                    capture_id=f"{application.value}-{index}.pcap",
                    vpn_status=VpnStatus.VPN if index % 2 else VpnStatus.NON_VPN,
                    application=application,
                    category=APPLICATION_TO_CATEGORY[application],
                    windows=25 + index,
                )
            )
    return captures


def test_default_configuration_contract() -> None:
    config = CaptureSplitConfig()

    assert config.fractions == {
        DatasetPartition.TRAIN: 0.60,
        DatasetPartition.VALIDATION: 0.15,
        DatasetPartition.CALIBRATION: 0.10,
        DatasetPartition.TEST: 0.15,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"train_fraction": 0.0}, "between zero and one"),
        ({"train_fraction": 1.0}, "between zero and one"),
        ({"train_fraction": 0.50}, "sum to one"),
        ({"minimum_category_windows": 0}, "must be positive"),
        ({"relative_mip_gap": -0.1}, "MIP gap must be between"),
        ({"relative_mip_gap": 1.0}, "MIP gap must be between"),
        ({"solver_time_limit_seconds": 0.0}, "time limit must be positive"),
    ],
)
def test_rejects_invalid_configuration(changes: dict[str, float | int], message: str) -> None:
    with pytest.raises(CaptureSplitError, match=message):
        CaptureSplitConfig(**changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("captures", "message"),
    [
        ([], "at least one"),
        (
            [
                CaptureGroup(
                    "",
                    VpnStatus.VPN,
                    Application.VOIP,
                    TrafficCategory.VOIP,
                    25,
                )
            ],
            "must not be empty",
        ),
        (
            [
                CaptureGroup(
                    "duplicate.pcap",
                    VpnStatus.VPN,
                    Application.VOIP,
                    TrafficCategory.VOIP,
                    25,
                ),
                CaptureGroup(
                    "duplicate.pcap",
                    VpnStatus.NON_VPN,
                    Application.VOIP,
                    TrafficCategory.VOIP,
                    25,
                ),
            ],
            "must be unique",
        ),
        (
            [
                CaptureGroup(
                    "empty.pcap",
                    VpnStatus.VPN,
                    Application.VOIP,
                    TrafficCategory.VOIP,
                    0,
                )
            ],
            "must be positive",
        ),
        (
            [
                CaptureGroup(
                    "mismatch.pcap",
                    VpnStatus.VPN,
                    Application.VOIP,
                    TrafficCategory.CHAT,
                    25,
                )
            ],
            "application/category mismatch",
        ),
    ],
)
def test_rejects_invalid_capture_groups(captures: list[CaptureGroup], message: str) -> None:
    with pytest.raises(CaptureSplitError, match=message):
        assign_capture_splits(captures)


def test_assigns_deterministic_leakage_free_capture_splits() -> None:
    captures = feasible_captures()

    first = assign_capture_splits(captures)
    second = assign_capture_splits(reversed(captures))

    assert first.assignments == second.assignments
    assert first.objective_value == pytest.approx(second.objective_value)
    assert first.optimal
    assert "Optimal" in first.solver_message
    assert len(first.assignments) == len(captures)
    assert len({item.capture.capture_id for item in first.assignments}) == len(captures)

    for category in TrafficCategory:
        for partition in DATASET_PARTITIONS:
            selected = [
                item
                for item in first.assignments
                if item.capture.category is category and item.partition is partition
            ]
            assert selected
            assert sum(item.capture.windows for item in selected) >= 20

    training = [item for item in first.assignments if item.partition is DatasetPartition.TRAIN]
    for partition in (DatasetPartition.TRAIN, DatasetPartition.TEST):
        assert {
            item.capture.application for item in first.assignments if item.partition is partition
        } == set(Application)
    for category in TrafficCategory:
        assert {
            item.capture.vpn_status for item in training if item.capture.category is category
        } == set(VpnStatus)

    for partition in DATASET_PARTITIONS:
        assert {
            item.capture.vpn_status for item in first.assignments if item.partition is partition
        } == set(VpnStatus)

    known_capture = first.assignments[0]
    assert first.partition_for(known_capture.capture.capture_id) is known_capture.partition
    with pytest.raises(KeyError, match=r"unknown\.pcap"):
        first.partition_for("unknown.pcap")


def test_reports_infeasible_coverage_contract() -> None:
    captures = [
        capture for capture in feasible_captures() if capture.category is not TrafficCategory.VOIP
    ]

    with pytest.raises(CaptureSplitError, match="optimization failed"):
        assign_capture_splits(captures)


def test_rejects_solver_output_without_exactly_one_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_result(**kwargs: object) -> SimpleNamespace:
        objective = kwargs["c"]
        assert isinstance(objective, np.ndarray)
        return SimpleNamespace(
            success=True,
            x=np.zeros_like(objective),
            fun=0.0,
            message="invalid test result",
        )

    monkeypatch.setattr(splitting, "milp", invalid_result)

    with pytest.raises(CaptureSplitError, match="invalid capture assignment"):
        assign_capture_splits(feasible_captures())


def test_validator_rejects_missing_category_partition() -> None:
    assignments = assign_capture_splits(feasible_captures()).assignments
    invalid = tuple(
        item
        for item in assignments
        if not (
            item.capture.category is TrafficCategory.VOIP
            and item.partition is DatasetPartition.TEST
        )
    )

    with pytest.raises(CaptureSplitError, match="cover every category partition"):
        validate_capture_assignments(invalid)


def test_validator_rejects_category_below_window_floor() -> None:
    assignments = assign_capture_splits(feasible_captures()).assignments
    invalid = tuple(
        replace(item, capture=replace(item.capture, windows=1))
        if item.capture.category is TrafficCategory.VOIP and item.partition is DatasetPartition.TEST
        else item
        for item in assignments
    )

    with pytest.raises(CaptureSplitError, match="category window floor"):
        validate_capture_assignments(invalid)


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        (DatasetPartition.TRAIN, DatasetPartition.VALIDATION),
        (DatasetPartition.TEST, DatasetPartition.VALIDATION),
    ],
)
def test_validator_rejects_missing_required_application(
    source: DatasetPartition, destination: DatasetPartition
) -> None:
    assignments = assign_capture_splits(feasible_captures()).assignments
    invalid = tuple(
        replace(item, partition=destination)
        if item.capture.application is Application.VIMEO and item.partition is source
        else item
        for item in assignments
    )

    with pytest.raises(CaptureSplitError, match="every training and test application"):
        validate_capture_assignments(
            invalid,
            config=replace(CaptureSplitConfig(), minimum_category_windows=1),
        )


def test_validator_rejects_partition_missing_vpn_status() -> None:
    assignments = assign_capture_splits(feasible_captures()).assignments
    invalid = tuple(
        replace(item, capture=replace(item.capture, vpn_status=VpnStatus.NON_VPN))
        if item.partition is DatasetPartition.TEST
        else item
        for item in assignments
    )

    with pytest.raises(CaptureSplitError, match="both VPN statuses"):
        validate_capture_assignments(invalid)


def test_validator_rejects_training_category_missing_vpn_status() -> None:
    assignments = assign_capture_splits(feasible_captures()).assignments
    invalid = tuple(
        replace(item, capture=replace(item.capture, vpn_status=VpnStatus.NON_VPN))
        if item.partition is DatasetPartition.TRAIN and item.capture.category is TrafficCategory.C2
        else item
        for item in assignments
    )

    with pytest.raises(CaptureSplitError, match="each category and VPN status"):
        validate_capture_assignments(invalid)
