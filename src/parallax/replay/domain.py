"""Immutable domain contracts for deterministic PCAP replay sessions."""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from re import compile as compile_pattern
from typing import Self
from uuid import UUID, uuid4


class ReplayDomainError(ValueError):
    """Raised when replay-domain state violates its contract."""


@dataclass(frozen=True, slots=True)
class ReplaySessionId:
    """Opaque UUID identity for one replay session."""

    value: UUID

    @classmethod
    def new(cls) -> Self:
        """Create a new replay-session identity."""
        return cls(uuid4())

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse a textual UUID into a replay-session identity."""
        try:
            return cls(UUID(value))
        except ValueError as error:
            raise ReplayDomainError("replay session ID must be a valid UUID") from error

    def __str__(self) -> str:
        """Return the canonical textual UUID."""
        return str(self.value)


class ReplayState(StrEnum):
    """Lifecycle state for one replay session."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_REPLAY_STATES = frozenset(
    {
        ReplayState.COMPLETED,
        ReplayState.FAILED,
        ReplayState.CANCELLED,
    }
)

_VALID_REPLAY_TRANSITIONS: dict[ReplayState, frozenset[ReplayState]] = {
    ReplayState.CREATED: frozenset(
        {
            ReplayState.RUNNING,
            ReplayState.FAILED,
            ReplayState.CANCELLED,
        }
    ),
    ReplayState.RUNNING: frozenset(
        {
            ReplayState.PAUSED,
            ReplayState.COMPLETED,
            ReplayState.FAILED,
            ReplayState.CANCELLED,
        }
    ),
    ReplayState.PAUSED: frozenset(
        {
            ReplayState.RUNNING,
            ReplayState.FAILED,
            ReplayState.CANCELLED,
        }
    ),
    ReplayState.COMPLETED: frozenset(),
    ReplayState.FAILED: frozenset(),
    ReplayState.CANCELLED: frozenset(),
}


def allowed_replay_transitions(state: ReplayState) -> frozenset[ReplayState]:
    """Return the immutable set of valid next states."""
    return _VALID_REPLAY_TRANSITIONS[state]


@dataclass(frozen=True, slots=True)
class ReplayConfiguration:
    """Scheduling configuration independent of the replay clock implementation."""

    time_scale: float | None = 1.0

    def __post_init__(self) -> None:
        if self.time_scale is None:
            return
        if not isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ReplayDomainError("replay time scale must be finite and greater than zero")

    @property
    def maximum_speed(self) -> bool:
        """Return whether replay should run without wall-clock delay."""
        return self.time_scale is None


@dataclass(frozen=True, slots=True)
class ReplayFailure:
    """Structured terminal failure retained with a failed replay session."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ReplayDomainError("replay failure code must not be empty")
        if not self.message.strip():
            raise ReplayDomainError("replay failure message must not be empty")


_SHA256_PATTERN = compile_pattern(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ReplaySession:
    """Immutable identity, configuration, and lifecycle state for one replay."""

    session_id: ReplaySessionId
    source_id: str
    source_sha256: str
    configuration: ReplayConfiguration
    state: ReplayState = ReplayState.CREATED
    failure: ReplayFailure | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ReplayDomainError("replay source ID must not be empty")
        if _SHA256_PATTERN.fullmatch(self.source_sha256) is None:
            raise ReplayDomainError("replay source SHA-256 must be 64 lowercase hexadecimal digits")
        if self.state is ReplayState.FAILED:
            if self.failure is None:
                raise ReplayDomainError("failed replay sessions must retain a failure")
        elif self.failure is not None:
            raise ReplayDomainError("only failed replay sessions may retain a failure")

    @property
    def is_terminal(self) -> bool:
        """Return whether no further lifecycle transitions are permitted."""
        return self.state in _TERMINAL_REPLAY_STATES

    def transition(
        self,
        target: ReplayState,
        *,
        failure: ReplayFailure | None = None,
    ) -> Self:
        """Return a new session in a valid next lifecycle state."""
        if target not in allowed_replay_transitions(self.state):
            raise ReplayDomainError(
                f"invalid replay transition: {self.state.value} -> {target.value}"
            )
        if target is ReplayState.FAILED:
            if failure is None:
                raise ReplayDomainError("transition to failed requires a replay failure")
        elif failure is not None:
            raise ReplayDomainError("failure details are only valid for transition to failed")

        return replace(self, state=target, failure=failure)
