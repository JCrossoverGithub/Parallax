from collections.abc import Iterator

import pytest

from parallax.replay import (
    ReplayConfiguration,
    ReplayFailure,
    ReplayRunnerError,
    ReplayScheduleEntry,
    ReplaySession,
    ReplaySessionId,
    ReplayState,
    run_replay_schedule,
)

SESSION_ID = "4d524cee-1288-45d7-9c6f-fdbab196bb26"
SOURCE_SHA256 = "a" * 64


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


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


def test_runner_completes_created_session_and_handles_entries_in_order() -> None:
    clock = _FakeClock()
    handled: list[int] = []

    completed = run_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 2.0), _entry(3, 5.0)],
        handle_entry=lambda entry: handled.append(entry.packet_number),
        clock=clock,
    )

    assert completed.state is ReplayState.COMPLETED
    assert completed.failure is None
    assert completed.is_terminal is True
    assert handled == [1, 2, 3]
    assert clock.sleep_calls == [2.0, 3.0]


def test_runner_handles_empty_schedule_as_successful_completion() -> None:
    handled: list[int] = []

    completed = run_replay_schedule(
        _session(),
        [],
        handle_entry=lambda entry: handled.append(entry.packet_number),
        clock=_FakeClock(),
    )

    assert completed.state is ReplayState.COMPLETED
    assert handled == []


@pytest.mark.parametrize(
    "session",
    [
        _session().transition(ReplayState.RUNNING),
        _session().transition(ReplayState.RUNNING).transition(ReplayState.PAUSED),
        _session().transition(ReplayState.RUNNING).transition(ReplayState.COMPLETED),
        _session().transition(ReplayState.CANCELLED),
    ],
)
def test_runner_requires_created_session(session: ReplaySession) -> None:
    with pytest.raises(ReplayRunnerError, match="requires a created session"):
        run_replay_schedule(
            session,
            [],
            handle_entry=lambda entry: None,
            clock=_FakeClock(),
        )


def test_runner_rejects_preexisting_failed_session() -> None:
    failed = _session().transition(
        ReplayState.FAILED,
        failure=ReplayFailure(code="fixture_failure", message="already failed"),
    )

    with pytest.raises(ReplayRunnerError, match="requires a created session"):
        run_replay_schedule(
            failed,
            [],
            handle_entry=lambda entry: None,
            clock=_FakeClock(),
        )


def test_handler_exception_becomes_structured_failed_session() -> None:
    handled: list[int] = []

    def handle(entry: ReplayScheduleEntry) -> None:
        handled.append(entry.packet_number)
        if entry.packet_number == 2:
            raise RuntimeError("consumer failed")

    failed = run_replay_schedule(
        _session(),
        [_entry(1, 0.0), _entry(2, 1.0), _entry(3, 2.0)],
        handle_entry=handle,
        clock=_FakeClock(),
    )

    assert failed.state is ReplayState.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "replay_execution_error"
    assert failed.failure.message == "RuntimeError: consumer failed"
    assert handled == [1, 2]


def test_schedule_exception_becomes_structured_failed_session() -> None:
    def schedule() -> Iterator[ReplayScheduleEntry]:
        yield _entry(1, 0.0)
        raise OSError("source disappeared")

    handled: list[int] = []

    failed = run_replay_schedule(
        _session(),
        schedule(),
        handle_entry=lambda entry: handled.append(entry.packet_number),
        clock=_FakeClock(),
    )

    assert failed.state is ReplayState.FAILED
    assert failed.failure is not None
    assert failed.failure.message == "OSError: source disappeared"
    assert handled == [1]


def test_pacing_exception_becomes_structured_failed_session() -> None:
    failed = run_replay_schedule(
        _session(),
        [_entry(1, -1.0)],
        handle_entry=lambda entry: None,
        clock=_FakeClock(),
    )

    assert failed.state is ReplayState.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "replay_execution_error"
    assert "ReplayClockError" in failed.failure.message


def test_exception_without_message_still_produces_valid_failure() -> None:
    def fail(_: ReplayScheduleEntry) -> None:
        raise RuntimeError()

    failed = run_replay_schedule(
        _session(),
        [_entry(1, 0.0)],
        handle_entry=fail,
        clock=_FakeClock(),
    )

    assert failed.failure is not None
    assert failed.failure.message == "RuntimeError"


def test_runner_consumes_schedule_lazily() -> None:
    def schedule() -> Iterator[ReplayScheduleEntry]:
        yield _entry(1, 0.0)
        raise RuntimeError("later entry requested")

    handled: list[int] = []

    failed = run_replay_schedule(
        _session(),
        schedule(),
        handle_entry=lambda entry: handled.append(entry.packet_number),
        clock=_FakeClock(),
    )

    assert handled == [1]
    assert failed.state is ReplayState.FAILED
