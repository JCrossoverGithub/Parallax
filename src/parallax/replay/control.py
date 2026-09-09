"""Externally mutable control state for one running replay."""

from enum import StrEnum
from threading import Lock

from parallax.replay.domain import ReplayDomainError


class ReplayControlError(ReplayDomainError):
    """Raised when a replay-control request is invalid."""


class ReplayControlState(StrEnum):
    """Requested execution state for a replay runner."""

    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ReplayControl:
    """Thread-safe external pause, resume, and cancellation requests."""

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = ReplayControlState.RUNNING

    @property
    def state(self) -> ReplayControlState:
        """Return the current requested execution state."""
        with self._lock:
            return self._state

    @property
    def is_paused(self) -> bool:
        """Return whether replay execution is currently requested to pause."""
        return self.state is ReplayControlState.PAUSED

    @property
    def is_cancelled(self) -> bool:
        """Return whether replay execution has been permanently cancelled."""
        return self.state is ReplayControlState.CANCELLED

    def pause(self) -> None:
        """Request paused execution; repeated pause requests are idempotent."""
        with self._lock:
            if self._state is ReplayControlState.CANCELLED:
                raise ReplayControlError("cancelled replay control cannot be paused")
            self._state = ReplayControlState.PAUSED

    def resume(self) -> None:
        """Request running execution; repeated resume requests are idempotent."""
        with self._lock:
            if self._state is ReplayControlState.CANCELLED:
                raise ReplayControlError("cancelled replay control cannot be resumed")
            self._state = ReplayControlState.RUNNING

    def cancel(self) -> None:
        """Permanently request cancellation; repeated requests are idempotent."""
        with self._lock:
            self._state = ReplayControlState.CANCELLED
