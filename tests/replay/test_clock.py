from collections.abc import Iterator
from math import inf, isfinite, nan

import pytest

from parallax.replay import (
    ReplayClockError,
    ReplayDomainError,
    ReplayScheduleEntry,
    SystemReplayClock,
    iter_paced_replay_schedule,
    wait_until_replay_offset,
)


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


def _entry(packet_number: int, offset: float) -> ReplayScheduleEntry:
    return ReplayScheduleEntry(
        packet_number=packet_number,
        source_timestamp_seconds=1_000.0 + offset,
        source_offset_seconds=offset,
        scheduled_offset_seconds=offset,
    )


def test_system_replay_clock_exposes_monotonic_time_and_zero_sleep() -> None:
    clock = SystemReplayClock()

    before = clock.monotonic()
    clock.sleep(0.0)
    after = clock.monotonic()

    assert isfinite(before)
    assert after >= before


def test_wait_until_replay_offset_sleeps_until_absolute_deadline() -> None:
    clock = _FakeClock(now=100.0)

    wait_until_replay_offset(
        2.5,
        replay_started_at=100.0,
        clock=clock,
    )

    assert clock.sleep_calls == [2.5]
    assert clock.now == 102.5


def test_wait_until_replay_offset_does_not_sleep_when_already_due() -> None:
    clock = _FakeClock(now=105.0)

    wait_until_replay_offset(
        2.5,
        replay_started_at=100.0,
        clock=clock,
    )

    assert clock.sleep_calls == []


class _EarlyWakeClock(_FakeClock):
    def __init__(self) -> None:
        super().__init__(now=100.0)
        self._early_wakeup = True

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        if self._early_wakeup:
            self.now += seconds / 2.0
            self._early_wakeup = False
        else:
            self.now += seconds


def test_wait_rechecks_clock_after_early_wakeup() -> None:
    clock = _EarlyWakeClock()

    wait_until_replay_offset(
        4.0,
        replay_started_at=100.0,
        clock=clock,
    )

    assert clock.sleep_calls == [4.0, 2.0]
    assert clock.now == 104.0


@pytest.mark.parametrize("replay_started_at", [nan, inf, -inf])
def test_wait_rejects_nonfinite_replay_start(replay_started_at: float) -> None:
    with pytest.raises(ReplayClockError, match="start time must be finite"):
        wait_until_replay_offset(
            1.0,
            replay_started_at=replay_started_at,
            clock=_FakeClock(),
        )


@pytest.mark.parametrize("offset", [-1.0, nan, inf, -inf])
def test_wait_rejects_invalid_scheduled_offset(offset: float) -> None:
    with pytest.raises(ReplayClockError, match="finite and nonnegative"):
        wait_until_replay_offset(
            offset,
            replay_started_at=100.0,
            clock=_FakeClock(),
        )


def test_wait_rejects_nonfinite_computed_deadline() -> None:
    with pytest.raises(ReplayClockError, match="deadline must be finite"):
        wait_until_replay_offset(
            1.0e308,
            replay_started_at=1.0e308,
            clock=_FakeClock(),
        )


class _NonfiniteClock:
    def monotonic(self) -> float:
        return nan

    def sleep(self, seconds: float) -> None:
        raise AssertionError(f"unexpected sleep: {seconds}")


def test_wait_rejects_nonfinite_clock_value() -> None:
    with pytest.raises(ReplayClockError, match="clock must return finite"):
        wait_until_replay_offset(
            1.0,
            replay_started_at=0.0,
            clock=_NonfiniteClock(),
        )


class _BackwardClock:
    def __init__(self) -> None:
        self.values = iter([1.0, 0.5])

    def monotonic(self) -> float:
        return next(self.values)

    def sleep(self, seconds: float) -> None:
        assert seconds == 1.0


def test_wait_rejects_clock_that_moves_backward() -> None:
    with pytest.raises(ReplayClockError, match="must not move backward"):
        wait_until_replay_offset(
            2.0,
            replay_started_at=0.0,
            clock=_BackwardClock(),
        )


def test_clock_errors_are_replay_domain_errors() -> None:
    with pytest.raises(ReplayDomainError):
        wait_until_replay_offset(
            -1.0,
            replay_started_at=0.0,
            clock=_FakeClock(),
        )


def test_paced_schedule_uses_absolute_offsets_without_cumulative_drift() -> None:
    class OversleepClock(_FakeClock):
        def sleep(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now += seconds + 0.5

    clock = OversleepClock(now=100.0)
    schedule = [_entry(1, 0.0), _entry(2, 2.0), _entry(3, 5.0)]

    result = list(iter_paced_replay_schedule(schedule, clock=clock))

    assert result == schedule
    assert clock.sleep_calls == [2.0, 2.5]
    assert clock.now == 105.5


def test_paced_schedule_is_lazy() -> None:
    clock = _FakeClock(now=100.0)

    def schedule() -> Iterator[ReplayScheduleEntry]:
        yield _entry(1, 0.0)
        raise RuntimeError("later schedule entry requested")

    paced = iter_paced_replay_schedule(schedule(), clock=clock)

    assert next(paced) == _entry(1, 0.0)

    with pytest.raises(RuntimeError, match="later schedule entry requested"):
        next(paced)


def test_empty_paced_schedule_does_not_read_clock() -> None:
    class UnreadableClock:
        def monotonic(self) -> float:
            raise AssertionError("clock should not be read")

        def sleep(self, seconds: float) -> None:
            raise AssertionError(f"unexpected sleep: {seconds}")

    assert list(iter_paced_replay_schedule([], clock=UnreadableClock())) == []


def test_default_paced_schedule_can_yield_immediately_due_entry() -> None:
    entry = _entry(1, 0.0)

    assert list(iter_paced_replay_schedule([entry])) == [entry]
