"""Operator-domain contracts for controlled live sensor sessions."""

from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite

from parallax.data import RuntimeFlowTrackerConfig


class OperatorLiveSessionError(ValueError):
    """Raised when a live operator session is invalid."""


class OperatorLiveState(StrEnum):
    """Lifecycle state for one operator-controlled live sensor session."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Return whether this state permanently ends the live session."""
        return self in {
            OperatorLiveState.COMPLETED,
            OperatorLiveState.FAILED,
        }


@dataclass(frozen=True, slots=True)
class OperatorLiveConfiguration:
    """Explicit operator configuration for one live sensor session."""

    interface: str
    stale_after_seconds: float
    max_tracked_flows: int

    def __post_init__(self) -> None:
        if not self.interface.strip():
            raise OperatorLiveSessionError("live capture interface must not be empty")

        if not isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0.0:
            raise OperatorLiveSessionError("stale flow timeout must be finite and positive")

        if self.max_tracked_flows < 1:
            raise OperatorLiveSessionError("maximum tracked flows must be positive")

    def flow_config(self) -> RuntimeFlowTrackerConfig:
        """Build the runtime resource policy represented by this session."""
        return RuntimeFlowTrackerConfig(
            stale_after_seconds=self.stale_after_seconds,
            max_tracked_flows=self.max_tracked_flows,
        )


@dataclass(frozen=True, slots=True)
class OperatorLiveFailure:
    """Structured terminal failure for one live sensor session."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise OperatorLiveSessionError("live failure code must not be empty")

        if not self.message.strip():
            raise OperatorLiveSessionError("live failure message must not be empty")

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible failure representation."""
        return {
            "code": self.code,
            "message": self.message,
        }


_NORMAL_TRANSITIONS: dict[
    OperatorLiveState,
    frozenset[OperatorLiveState],
] = {
    OperatorLiveState.STARTING: frozenset(
        {
            OperatorLiveState.RUNNING,
            OperatorLiveState.STOPPING,
        }
    ),
    OperatorLiveState.RUNNING: frozenset(
        {
            OperatorLiveState.STOPPING,
        }
    ),
    OperatorLiveState.STOPPING: frozenset(
        {
            OperatorLiveState.COMPLETED,
        }
    ),
    OperatorLiveState.COMPLETED: frozenset(),
    OperatorLiveState.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class OperatorLiveSession:
    """Immutable lifecycle state for one operator-controlled live capture."""

    run_id: str
    configuration: OperatorLiveConfiguration
    state: OperatorLiveState = OperatorLiveState.STARTING
    failure: OperatorLiveFailure | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise OperatorLiveSessionError("live session run ID must not be empty")

        if self.state is OperatorLiveState.FAILED and self.failure is None:
            raise OperatorLiveSessionError("failed live session must include failure details")

        if self.state is not OperatorLiveState.FAILED and self.failure is not None:
            raise OperatorLiveSessionError(
                "nonfailed live session must not include failure details"
            )

    def transition(
        self,
        state: OperatorLiveState,
    ) -> "OperatorLiveSession":
        """Return the next valid nonfailure lifecycle state."""
        if state not in _NORMAL_TRANSITIONS[self.state]:
            raise OperatorLiveSessionError(
                f"invalid live session transition: {self.state.value} -> {state.value}"
            )

        return replace(
            self,
            state=state,
        )

    def fail(
        self,
        *,
        code: str,
        message: str,
    ) -> "OperatorLiveSession":
        """Return the terminal failed form of a nonterminal session."""
        if self.state.is_terminal:
            raise OperatorLiveSessionError(f"live session is already terminal: {self.state.value}")

        return replace(
            self,
            state=OperatorLiveState.FAILED,
            failure=OperatorLiveFailure(
                code=code,
                message=message,
            ),
        )
