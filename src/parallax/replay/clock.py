"""Clock-driven pacing primitives for deterministic replay schedules."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from time import monotonic, sleep
from typing import Protocol

from parallax.replay.control import ReplayControl, ReplayControlState
from parallax.replay.domain import ReplayDomainError
from parallax.replay.timing import ReplayScheduleEntry


class ReplayClockError(ReplayDomainError):
    """Raised when a replay clock cannot provide a valid monotonic timeline."""


class ReplayClock(Protocol):
    """Clock interface used to make replay pacing deterministic in tests."""

    def monotonic(self) -> float:
        """Return the current monotonic clock value."""

    def sleep(self, seconds: float) -> None:
        """Block for up to the requested positive duration."""


@dataclass(frozen=True, slots=True)
class SystemReplayClock:
    """Production clock backed by Python's monotonic clock and sleep."""

    def monotonic(self) -> float:
        """Return the operating system monotonic clock."""
        return monotonic()

    def sleep(self, seconds: float) -> None:
        """Sleep for the requested duration."""
        sleep(seconds)


def wait_until_replay_offset(
    scheduled_offset_seconds: float,
    *,
    replay_started_at: float,
    clock: ReplayClock,
) -> None:
    """Wait until one absolute replay offset without accumulating prior delay."""
    if not isfinite(replay_started_at):
        raise ReplayClockError("replay start time must be finite")
    if not isfinite(scheduled_offset_seconds) or scheduled_offset_seconds < 0.0:
        raise ReplayClockError("scheduled replay offset must be finite and nonnegative")

    deadline = replay_started_at + scheduled_offset_seconds
    if not isfinite(deadline):
        raise ReplayClockError("replay deadline must be finite")

    previous_now: float | None = None

    while True:
        now = clock.monotonic()
        if not isfinite(now):
            raise ReplayClockError("replay clock must return finite values")
        if previous_now is not None and now < previous_now:
            raise ReplayClockError("replay clock must not move backward")

        remaining = deadline - now
        if remaining <= 0.0:
            return

        previous_now = now
        clock.sleep(remaining)


def iter_paced_replay_schedule(
    schedule: Iterable[ReplayScheduleEntry],
    *,
    clock: ReplayClock | None = None,
) -> Iterator[ReplayScheduleEntry]:
    """Yield schedule entries only after their absolute replay offsets are due."""
    active_clock = clock if clock is not None else SystemReplayClock()
    replay_started_at: float | None = None

    for entry in schedule:
        if replay_started_at is None:
            replay_started_at = active_clock.monotonic()

        wait_until_replay_offset(
            entry.scheduled_offset_seconds,
            replay_started_at=replay_started_at,
            clock=active_clock,
        )
        yield entry


@dataclass(frozen=True, slots=True)
class ReplayPacingState:
    """Wall-clock origin and accumulated observed pause time for one replay."""

    replay_started_at: float
    paused_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.replay_started_at):
            raise ReplayClockError("replay start time must be finite")
        if not isfinite(self.paused_seconds) or self.paused_seconds < 0.0:
            raise ReplayClockError("paused replay time must be finite and nonnegative")


class ReplayWaitOutcome(StrEnum):
    """Result of one interruptible replay wait."""

    DUE = "due"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ReplayWaitResult:
    """Outcome and updated pacing state after one interruptible wait."""

    outcome: ReplayWaitOutcome
    pacing_state: ReplayPacingState


def wait_until_controlled_replay_offset(
    scheduled_offset_seconds: float,
    *,
    pacing_state: ReplayPacingState,
    clock: ReplayClock,
    control: ReplayControl,
    poll_interval_seconds: float = 0.1,
) -> ReplayWaitResult:
    """Wait responsively for one replay deadline while honoring pause and cancel.

    Control transitions are observed at polling boundaries. Cancellation
    responsiveness and pause-time accounting are therefore bounded by the
    configured polling interval.
    """
    if not isfinite(scheduled_offset_seconds) or scheduled_offset_seconds < 0.0:
        raise ReplayClockError("scheduled replay offset must be finite and nonnegative")
    if not isfinite(poll_interval_seconds) or poll_interval_seconds <= 0.0:
        raise ReplayClockError("control poll interval must be finite and greater than zero")

    paused_seconds = pacing_state.paused_seconds
    previous_now: float | None = None
    paused_at: float | None = None

    while True:
        now = clock.monotonic()
        if not isfinite(now):
            raise ReplayClockError("replay clock must return finite values")
        if previous_now is not None and now < previous_now:
            raise ReplayClockError("replay clock must not move backward")

        if paused_at is not None:
            paused_seconds += now - paused_at
            paused_at = None

        control_state = control.state

        if control_state is ReplayControlState.CANCELLED:
            return ReplayWaitResult(
                outcome=ReplayWaitOutcome.CANCELLED,
                pacing_state=ReplayPacingState(
                    replay_started_at=pacing_state.replay_started_at,
                    paused_seconds=paused_seconds,
                ),
            )

        if control_state is ReplayControlState.PAUSED:
            paused_at = now
            previous_now = now
            clock.sleep(poll_interval_seconds)
            continue

        deadline = pacing_state.replay_started_at + paused_seconds + scheduled_offset_seconds
        if not isfinite(deadline):
            raise ReplayClockError("replay deadline must be finite")

        remaining = deadline - now
        if remaining <= 0.0:
            return ReplayWaitResult(
                outcome=ReplayWaitOutcome.DUE,
                pacing_state=ReplayPacingState(
                    replay_started_at=pacing_state.replay_started_at,
                    paused_seconds=paused_seconds,
                ),
            )

        previous_now = now
        clock.sleep(min(remaining, poll_interval_seconds))
