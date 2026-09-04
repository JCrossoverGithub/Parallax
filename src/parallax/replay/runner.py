"""Synchronous orchestration for one deterministic replay schedule."""

from collections.abc import Callable, Iterable

from parallax.replay.clock import ReplayClock, iter_paced_replay_schedule
from parallax.replay.domain import (
    ReplayDomainError,
    ReplayFailure,
    ReplaySession,
    ReplayState,
)
from parallax.replay.timing import ReplayScheduleEntry


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
