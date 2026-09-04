from collections.abc import Callable, Iterator

import pytest

from parallax.replay.control import ReplayControl
from parallax.replay.domain import (
    ReplayConfiguration,
    ReplayFailure,
    ReplaySession,
    ReplaySessionId,
    ReplayState,
)
from parallax.replay.runner import ReplayRunnerError, run_controlled_replay_schedule
from parallax.replay.timing import ReplayScheduleEntry

SESSION_ID = "4d524cee-1288-45d7-9c6f-fdbab196bb26"
SOURCE_SHA256 = "a" * 64


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


def _session() -> ReplaySession:
    return ReplaySession(
        session_id=ReplaySessionId.parse(SESSION_ID),
        source_id="synthetic-replay.pcap",
        source_sha256=SOURCE_SHA256,
        configuration=ReplayConfiguration(),
    )


def _entry(packet_number: int, offset: float) -> ReplayScheduleEntry:
    return ReplayScheduleEntry(
        packet_number=packet_number,
        source_timestamp_seconds=1_000.0 + offset,
        source_offset_seconds=offset,
        scheduled_offset_seconds=offset,
    )


def test_controlled_runner_completes_and_reports_transitions() -> None:
    observed: list[ReplayState] = []
    handled: list[int] = []

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 1.0)],
        handle_entry=lambda entry: handled.append(entry.packet_number),
        control=ReplayControl(),
        clock=_ScriptedClock(),
        poll_interval_seconds=0.5,
        handle_session=lambda session: observed.append(session.state),
    )

    assert result.state is ReplayState.COMPLETED
    assert handled == [1, 2]
    assert observed == [ReplayState.RUNNING, ReplayState.COMPLETED]


def test_controlled_runner_reflects_pause_and_resume() -> None:
    control = ReplayControl()
    observed: list[ReplayState] = []

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.pause()
        elif call_number == 3:
            control.resume()

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 3.0)],
        handle_entry=lambda entry: None,
        control=control,
        clock=_ScriptedClock(on_sleep=on_sleep),
        poll_interval_seconds=1.0,
        handle_session=lambda session: observed.append(session.state),
    )

    assert result.state is ReplayState.COMPLETED
    assert observed == [
        ReplayState.RUNNING,
        ReplayState.PAUSED,
        ReplayState.RUNNING,
        ReplayState.COMPLETED,
    ]


def test_controlled_runner_cancels_during_long_wait() -> None:
    control = ReplayControl()
    observed: list[ReplayState] = []
    handled: list[int] = []

    def on_sleep(_: int) -> None:
        control.cancel()

    clock = _ScriptedClock(on_sleep=on_sleep)

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 300.0)],
        handle_entry=lambda entry: handled.append(entry.packet_number),
        control=control,
        clock=clock,
        poll_interval_seconds=0.25,
        handle_session=lambda session: observed.append(session.state),
    )

    assert result.state is ReplayState.CANCELLED
    assert result.failure is None
    assert handled == [1]
    assert observed == [ReplayState.RUNNING, ReplayState.CANCELLED]
    assert clock.sleep_calls == [0.25]


def test_controlled_runner_can_cancel_while_paused() -> None:
    control = ReplayControl()
    observed: list[ReplayState] = []

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.pause()
        elif call_number == 2:
            control.cancel()

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 10.0)],
        handle_entry=lambda entry: None,
        control=control,
        clock=_ScriptedClock(on_sleep=on_sleep),
        poll_interval_seconds=0.5,
        handle_session=lambda session: observed.append(session.state),
    )

    assert result.state is ReplayState.CANCELLED
    assert observed == [
        ReplayState.RUNNING,
        ReplayState.PAUSED,
        ReplayState.CANCELLED,
    ]


def test_preexisting_cancellation_never_starts_replay() -> None:
    control = ReplayControl()
    control.cancel()
    observed: list[ReplayState] = []

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0)],
        handle_entry=lambda entry: None,
        control=control,
        clock=_ScriptedClock(),
        handle_session=lambda session: observed.append(session.state),
    )

    assert result.state is ReplayState.CANCELLED
    assert result.failure is None
    assert observed == [ReplayState.CANCELLED]


def test_handler_cancellation_does_not_consume_another_entry() -> None:
    control = ReplayControl()
    consumed: list[int] = []

    def schedule() -> Iterator[ReplayScheduleEntry]:
        consumed.append(1)
        yield _entry(1, 0.0)

        consumed.append(2)
        raise AssertionError("second schedule entry must not be consumed")

    def handle_entry(_: ReplayScheduleEntry) -> None:
        control.cancel()

    result = run_controlled_replay_schedule(
        _session(),
        schedule(),
        handle_entry=handle_entry,
        control=control,
        clock=_ScriptedClock(),
    )

    assert result.state is ReplayState.CANCELLED
    assert consumed == [1]


def test_cancellation_after_schedule_exhaustion_wins_over_completion() -> None:
    control = ReplayControl()

    class _CancelOnExhaustion:
        def __iter__(self) -> Iterator[ReplayScheduleEntry]:
            yield _entry(1, 0.0)
            control.cancel()

    result = run_controlled_replay_schedule(
        _session(),
        _CancelOnExhaustion(),
        handle_entry=lambda entry: None,
        control=control,
        clock=_ScriptedClock(),
    )

    assert result.state is ReplayState.CANCELLED


def test_controlled_runner_handler_failure_becomes_failed_session() -> None:
    def handle_entry(_: ReplayScheduleEntry) -> None:
        raise RuntimeError("consumer failed")

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0)],
        handle_entry=handle_entry,
        control=ReplayControl(),
        clock=_ScriptedClock(),
    )

    assert result.state is ReplayState.FAILED
    assert result.failure == ReplayFailure(
        code="replay_execution_error",
        message="RuntimeError: consumer failed",
    )


def test_session_handler_failure_becomes_failed_session() -> None:
    def handle_session(_: ReplaySession) -> None:
        raise RuntimeError("observer failed")

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0)],
        handle_entry=lambda entry: None,
        control=ReplayControl(),
        clock=_ScriptedClock(),
        handle_session=handle_session,
    )

    assert result.state is ReplayState.FAILED
    assert result.failure == ReplayFailure(
        code="replay_execution_error",
        message="RuntimeError: observer failed",
    )


def test_controlled_runner_requires_created_session() -> None:
    running = _session().transition(ReplayState.RUNNING)

    with pytest.raises(ReplayRunnerError, match="requires a created session"):
        run_controlled_replay_schedule(
            running,
            [],
            handle_entry=lambda entry: None,
            control=ReplayControl(),
            clock=_ScriptedClock(),
        )


def test_empty_schedule_completes_without_reading_clock() -> None:
    class _UnreadableClock:
        def monotonic(self) -> float:
            raise AssertionError("clock should not be read")

        def sleep(self, seconds: float) -> None:
            raise AssertionError(f"unexpected sleep: {seconds}")

    result = run_controlled_replay_schedule(
        _session(),
        [],
        handle_entry=lambda entry: None,
        control=ReplayControl(),
        clock=_UnreadableClock(),
    )

    assert result.state is ReplayState.COMPLETED


def test_controlled_runner_supports_default_system_clock() -> None:
    handled: list[int] = []

    result = run_controlled_replay_schedule(
        _session(),
        [_entry(1, 0.0)],
        handle_entry=lambda entry: handled.append(entry.packet_number),
        control=ReplayControl(),
    )

    assert result.state is ReplayState.COMPLETED
    assert handled == [1]
