from collections.abc import Iterator
from math import inf, nan

import pytest

from parallax.replay import (
    ReplayConfiguration,
    ReplayDomainError,
    ReplayScheduleEntry,
    ReplayTimingError,
    iter_replay_schedule,
)


def test_empty_timestamp_stream_produces_no_schedule() -> None:
    assert list(iter_replay_schedule([])) == []


def test_one_times_replay_preserves_source_offsets() -> None:
    schedule = list(iter_replay_schedule([100.0, 102.0, 105.5]))

    assert schedule == [
        ReplayScheduleEntry(
            packet_number=1,
            source_timestamp_seconds=100.0,
            source_offset_seconds=0.0,
            scheduled_offset_seconds=0.0,
        ),
        ReplayScheduleEntry(
            packet_number=2,
            source_timestamp_seconds=102.0,
            source_offset_seconds=2.0,
            scheduled_offset_seconds=2.0,
        ),
        ReplayScheduleEntry(
            packet_number=3,
            source_timestamp_seconds=105.5,
            source_offset_seconds=5.5,
            scheduled_offset_seconds=5.5,
        ),
    ]


@pytest.mark.parametrize(
    ("time_scale", "expected_offsets"),
    [
        (2.0, [0.0, 1.0, 2.75]),
        (0.5, [0.0, 4.0, 11.0]),
        (4.0, [0.0, 0.5, 1.375]),
    ],
)
def test_time_scale_changes_wall_clock_offsets_without_changing_source_offsets(
    time_scale: float,
    expected_offsets: list[float],
) -> None:
    schedule = list(
        iter_replay_schedule(
            [100.0, 102.0, 105.5],
            configuration=ReplayConfiguration(time_scale=time_scale),
        )
    )

    assert [entry.source_offset_seconds for entry in schedule] == [0.0, 2.0, 5.5]
    assert [entry.scheduled_offset_seconds for entry in schedule] == expected_offsets


def test_maximum_speed_preserves_source_offsets_without_scheduling_delay() -> None:
    schedule = list(
        iter_replay_schedule(
            [100.0, 102.0, 105.5],
            configuration=ReplayConfiguration(time_scale=None),
        )
    )

    assert [entry.source_offset_seconds for entry in schedule] == [0.0, 2.0, 5.5]
    assert [entry.scheduled_offset_seconds for entry in schedule] == [0.0, 0.0, 0.0]


def test_equal_source_timestamps_are_valid() -> None:
    schedule = list(iter_replay_schedule([10.0, 10.0, 10.25]))

    assert [entry.packet_number for entry in schedule] == [1, 2, 3]
    assert [entry.source_offset_seconds for entry in schedule] == [0.0, 0.0, 0.25]
    assert [entry.scheduled_offset_seconds for entry in schedule] == [0.0, 0.0, 0.25]


@pytest.mark.parametrize("timestamp", [nan, inf, -inf])
def test_rejects_nonfinite_source_timestamp(timestamp: float) -> None:
    with pytest.raises(ReplayTimingError, match="packet 1: source timestamp must be finite"):
        list(iter_replay_schedule([timestamp]))


def test_rejects_nonfinite_timestamp_after_valid_input() -> None:
    with pytest.raises(ReplayTimingError, match="packet 2: source timestamp must be finite"):
        list(iter_replay_schedule([1.0, nan]))


def test_rejects_decreasing_source_timestamps() -> None:
    with pytest.raises(
        ReplayTimingError,
        match="packet 3: source timestamp precedes the previous packet",
    ):
        list(iter_replay_schedule([10.0, 11.0, 10.5]))


def test_timing_errors_are_replay_domain_errors() -> None:
    with pytest.raises(ReplayDomainError):
        list(iter_replay_schedule([2.0, 1.0]))


def test_schedule_generation_is_lazy() -> None:
    def timestamps() -> Iterator[float]:
        yield 100.0
        raise RuntimeError("later timestamp requested")

    schedule = iter_replay_schedule(timestamps())

    assert next(schedule) == ReplayScheduleEntry(
        packet_number=1,
        source_timestamp_seconds=100.0,
        source_offset_seconds=0.0,
        scheduled_offset_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="later timestamp requested"):
        next(schedule)
