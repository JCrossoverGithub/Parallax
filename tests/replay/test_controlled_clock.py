from collections.abc import Callable
from math import inf, nan

import pytest

from parallax.replay.clock import (
    ReplayClockError,
    ReplayPacingState,
    ReplayWaitOutcome,
    wait_until_controlled_replay_offset,
)
from parallax.replay.control import ReplayControl


class _ScriptedClock:
    def __init__(
        self,
        *,
        now: float = 100.0,
        on_sleep: Callable[[int], None] | None = None,
    ) -> None:
        self.now = now
        self.sleep_calls: list[float] = []
        self._on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds
        if self._on_sleep is not None:
            self._on_sleep(len(self.sleep_calls))


def test_pacing_state_accepts_valid_origin_and_pause_time() -> None:
    state = ReplayPacingState(
        replay_started_at=100.0,
        paused_seconds=2.5,
    )

    assert state.replay_started_at == 100.0
    assert state.paused_seconds == 2.5


@pytest.mark.parametrize("started_at", [nan, inf, -inf])
def test_pacing_state_rejects_nonfinite_start(started_at: float) -> None:
    with pytest.raises(ReplayClockError, match="start time must be finite"):
        ReplayPacingState(replay_started_at=started_at)


@pytest.mark.parametrize("paused_seconds", [-1.0, nan, inf, -inf])
def test_pacing_state_rejects_invalid_paused_time(paused_seconds: float) -> None:
    with pytest.raises(ReplayClockError, match="paused replay time"):
        ReplayPacingState(
            replay_started_at=100.0,
            paused_seconds=paused_seconds,
        )


def test_controlled_wait_returns_due_without_pause() -> None:
    control = ReplayControl()
    clock = _ScriptedClock(now=100.0)

    result = wait_until_controlled_replay_offset(
        2.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=clock,
        control=control,
        poll_interval_seconds=0.5,
    )

    assert result.outcome is ReplayWaitOutcome.DUE
    assert result.pacing_state.paused_seconds == 0.0
    assert clock.sleep_calls == [0.5, 0.5, 0.5, 0.5]


def test_preexisting_cancellation_returns_without_sleeping() -> None:
    control = ReplayControl()
    control.cancel()
    clock = _ScriptedClock(now=100.0)

    result = wait_until_controlled_replay_offset(
        10.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=clock,
        control=control,
    )

    assert result.outcome is ReplayWaitOutcome.CANCELLED
    assert result.pacing_state.paused_seconds == 0.0
    assert clock.sleep_calls == []


def test_cancellation_interrupts_long_wait_at_poll_boundary() -> None:
    control = ReplayControl()

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.cancel()

    clock = _ScriptedClock(now=100.0, on_sleep=on_sleep)

    result = wait_until_controlled_replay_offset(
        300.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=clock,
        control=control,
        poll_interval_seconds=0.25,
    )

    assert result.outcome is ReplayWaitOutcome.CANCELLED
    assert clock.sleep_calls == [0.25]
    assert clock.now == 100.25


def test_pause_time_shifts_replay_deadline() -> None:
    control = ReplayControl()

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.pause()
        elif call_number == 3:
            control.resume()

    clock = _ScriptedClock(now=100.0, on_sleep=on_sleep)

    result = wait_until_controlled_replay_offset(
        3.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=clock,
        control=control,
        poll_interval_seconds=1.0,
    )

    assert result.outcome is ReplayWaitOutcome.DUE
    assert result.pacing_state.paused_seconds == 2.0
    assert clock.now == 105.0
    assert clock.sleep_calls == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_existing_pause_time_is_preserved_for_later_deadline() -> None:
    control = ReplayControl()
    clock = _ScriptedClock(now=104.0)

    result = wait_until_controlled_replay_offset(
        3.0,
        pacing_state=ReplayPacingState(
            replay_started_at=100.0,
            paused_seconds=2.0,
        ),
        clock=clock,
        control=control,
        poll_interval_seconds=1.0,
    )

    assert result.outcome is ReplayWaitOutcome.DUE
    assert result.pacing_state.paused_seconds == 2.0
    assert clock.now == 105.0
    assert clock.sleep_calls == [1.0]


@pytest.mark.parametrize("offset", [-1.0, nan, inf, -inf])
def test_controlled_wait_rejects_invalid_offset(offset: float) -> None:
    with pytest.raises(ReplayClockError, match="finite and nonnegative"):
        wait_until_controlled_replay_offset(
            offset,
            pacing_state=ReplayPacingState(replay_started_at=100.0),
            clock=_ScriptedClock(),
            control=ReplayControl(),
        )


@pytest.mark.parametrize("poll_interval", [0.0, -1.0, nan, inf, -inf])
def test_controlled_wait_rejects_invalid_poll_interval(
    poll_interval: float,
) -> None:
    with pytest.raises(ReplayClockError, match="poll interval"):
        wait_until_controlled_replay_offset(
            1.0,
            pacing_state=ReplayPacingState(replay_started_at=100.0),
            clock=_ScriptedClock(),
            control=ReplayControl(),
            poll_interval_seconds=poll_interval,
        )


def test_controlled_wait_rejects_nonfinite_clock_value() -> None:
    clock = _ScriptedClock(now=nan)

    with pytest.raises(ReplayClockError, match="clock must return finite"):
        wait_until_controlled_replay_offset(
            1.0,
            pacing_state=ReplayPacingState(replay_started_at=100.0),
            clock=clock,
            control=ReplayControl(),
        )


class _BackwardClock:
    def __init__(self) -> None:
        self._values = iter([100.0, 99.0])

    def monotonic(self) -> float:
        return next(self._values)

    def sleep(self, seconds: float) -> None:
        assert seconds == 0.5


def test_controlled_wait_rejects_backward_clock() -> None:
    with pytest.raises(ReplayClockError, match="must not move backward"):
        wait_until_controlled_replay_offset(
            2.0,
            pacing_state=ReplayPacingState(replay_started_at=100.0),
            clock=_BackwardClock(),
            control=ReplayControl(),
            poll_interval_seconds=0.5,
        )


def test_controlled_wait_rejects_nonfinite_deadline() -> None:
    with pytest.raises(ReplayClockError, match="deadline must be finite"):
        wait_until_controlled_replay_offset(
            1.0e308,
            pacing_state=ReplayPacingState(
                replay_started_at=1.0e308,
                paused_seconds=1.0e308,
            ),
            clock=_ScriptedClock(now=1.0e308),
            control=ReplayControl(),
        )


def test_controlled_wait_reports_pause_and_resume_transitions() -> None:
    control = ReplayControl()
    observed: list[str] = []

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.pause()
        elif call_number == 2:
            control.resume()

    clock = _ScriptedClock(now=100.0, on_sleep=on_sleep)

    result = wait_until_controlled_replay_offset(
        2.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=clock,
        control=control,
        poll_interval_seconds=1.0,
        handle_control_state=lambda state: observed.append(state.value),
    )

    assert result.outcome is ReplayWaitOutcome.DUE
    assert observed == ["paused", "running"]


def test_controlled_wait_reports_cancellation_once() -> None:
    control = ReplayControl()
    observed: list[str] = []

    def on_sleep(_: int) -> None:
        control.cancel()

    result = wait_until_controlled_replay_offset(
        100.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=_ScriptedClock(now=100.0, on_sleep=on_sleep),
        control=control,
        poll_interval_seconds=0.25,
        handle_control_state=lambda state: observed.append(state.value),
    )

    assert result.outcome is ReplayWaitOutcome.CANCELLED
    assert observed == ["cancelled"]


def test_controlled_wait_does_not_report_unchanged_running_state() -> None:
    observed: list[str] = []

    result = wait_until_controlled_replay_offset(
        1.0,
        pacing_state=ReplayPacingState(replay_started_at=100.0),
        clock=_ScriptedClock(now=100.0),
        control=ReplayControl(),
        poll_interval_seconds=0.5,
        handle_control_state=lambda state: observed.append(state.value),
    )

    assert result.outcome is ReplayWaitOutcome.DUE
    assert observed == []
