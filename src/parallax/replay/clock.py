"""Clock-driven pacing primitives for deterministic replay schedules."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from time import monotonic, sleep
from typing import Protocol

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
