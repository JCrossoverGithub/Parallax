"""Synchronous orchestration for one deterministic replay schedule."""

from collections.abc import Callable, Iterable

from parallax.replay.clock import (
    ReplayClock,
    ReplayPacingState,
    ReplayWaitOutcome,
    SystemReplayClock,
    iter_paced_replay_schedule,
    wait_until_controlled_replay_offset,
)
from parallax.replay.control import ReplayControl, ReplayControlState
from parallax.replay.domain import (
    ReplayDomainError,
    ReplayFailure,
    ReplaySession,
    ReplayState,
)
from parallax.replay.timing import ReplayScheduleEntry, ReplayTimedEntry


class ReplayRunnerError(ReplayDomainError):
    """Raised when a replay session cannot be executed by the runner."""


ReplayEntryHandler = Callable[[ReplayScheduleEntry], None]


def run_replay_schedule(
    session: ReplaySession,
    schedule: Iterable[ReplayScheduleEntry],
    *,
    handle_entry: ReplayEntryHandler,
    clock: ReplayClock | None = None,
) -> ReplaySession:
    """Run one created session to completion or structured execution failure."""
    if session.state is not ReplayState.CREATED:
        raise ReplayRunnerError("replay runner requires a created session")

    running = session.transition(ReplayState.RUNNING)

    try:
        for entry in iter_paced_replay_schedule(schedule, clock=clock):
            handle_entry(entry)
    except Exception as error:
        failure = ReplayFailure(
            code="replay_execution_error",
            message=_failure_message(error),
        )
        return running.transition(ReplayState.FAILED, failure=failure)

    return running.transition(ReplayState.COMPLETED)


def _failure_message(error: Exception) -> str:
    detail = str(error).strip()
    if detail:
        return f"{type(error).__name__}: {detail}"
    return type(error).__name__


ReplaySessionHandler = Callable[[ReplaySession], None]


def run_controlled_replay_schedule[ReplayTimedEntryT: ReplayTimedEntry](
    session: ReplaySession,
    schedule: Iterable[ReplayTimedEntryT],
    *,
    handle_entry: Callable[[ReplayTimedEntryT], None],
    control: ReplayControl,
    clock: ReplayClock | None = None,
    poll_interval_seconds: float = 0.1,
    handle_session: ReplaySessionHandler | None = None,
) -> ReplaySession:
    """Run a replay while reflecting pause, resume, and cancellation state."""
    if session.state is not ReplayState.CREATED:
        raise ReplayRunnerError("controlled replay runner requires a created session")

    current = session
    active_clock = clock if clock is not None else SystemReplayClock()

    try:
        if control.is_cancelled:
            return _transition_session(
                current,
                ReplayState.CANCELLED,
                handle_session=handle_session,
            )

        current = _transition_session(
            current,
            ReplayState.RUNNING,
            handle_session=handle_session,
        )
        pacing_state: ReplayPacingState | None = None

        def handle_control_state(state: ReplayControlState) -> None:
            nonlocal current

            if state is ReplayControlState.PAUSED:
                current = _transition_session(
                    current,
                    ReplayState.PAUSED,
                    handle_session=handle_session,
                )
            elif state is ReplayControlState.RUNNING:
                current = _transition_session(
                    current,
                    ReplayState.RUNNING,
                    handle_session=handle_session,
                )
            else:
                current = _transition_session(
                    current,
                    ReplayState.CANCELLED,
                    handle_session=handle_session,
                )

        for entry in schedule:
            if pacing_state is None:
                pacing_state = ReplayPacingState(replay_started_at=active_clock.monotonic())

            wait_result = wait_until_controlled_replay_offset(
                entry.scheduled_offset_seconds,
                pacing_state=pacing_state,
                clock=active_clock,
                control=control,
                poll_interval_seconds=poll_interval_seconds,
                handle_control_state=handle_control_state,
            )
            pacing_state = wait_result.pacing_state

            if wait_result.outcome is ReplayWaitOutcome.CANCELLED:
                return current

            handle_entry(entry)

            if control.is_cancelled:
                return _transition_session(
                    current,
                    ReplayState.CANCELLED,
                    handle_session=handle_session,
                )

        if control.is_cancelled:
            return _transition_session(
                current,
                ReplayState.CANCELLED,
                handle_session=handle_session,
            )

        return _transition_session(
            current,
            ReplayState.COMPLETED,
            handle_session=handle_session,
        )
    except Exception as error:
        failure = ReplayFailure(
            code="replay_execution_error",
            message=_failure_message(error),
        )
        return current.transition(ReplayState.FAILED, failure=failure)


def _transition_session(
    session: ReplaySession,
    target: ReplayState,
    *,
    handle_session: ReplaySessionHandler | None,
) -> ReplaySession:
    transitioned = session.transition(target)
    if handle_session is not None:
        handle_session(transitioned)
    return transitioned
