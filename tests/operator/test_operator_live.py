from math import inf, nan

import pytest

from parallax.data import RuntimeFlowTrackerConfig
from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveFailure,
    OperatorLiveSession,
    OperatorLiveSessionError,
    OperatorLiveState,
)


def _configuration() -> OperatorLiveConfiguration:
    return OperatorLiveConfiguration(
        interface="eth0",
        stale_after_seconds=120.0,
        max_tracked_flows=4_096,
    )


def test_live_configuration_builds_runtime_flow_policy() -> None:
    configuration = _configuration()

    assert configuration.interface == "eth0"

    assert configuration.flow_config() == RuntimeFlowTrackerConfig(
        stale_after_seconds=120.0,
        max_tracked_flows=4_096,
    )


@pytest.mark.parametrize("interface", ["", " ", "\t"])
def test_live_configuration_rejects_empty_interface(
    interface: str,
) -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="interface must not be empty",
    ):
        OperatorLiveConfiguration(
            interface=interface,
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
        )


@pytest.mark.parametrize(
    "stale_after_seconds",
    [0.0, -1.0, inf, -inf, nan],
)
def test_live_configuration_rejects_invalid_stale_timeout(
    stale_after_seconds: float,
) -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="stale flow timeout must be finite and positive",
    ):
        OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=stale_after_seconds,
            max_tracked_flows=4_096,
        )


@pytest.mark.parametrize("max_tracked_flows", [0, -1])
def test_live_configuration_rejects_invalid_flow_capacity(
    max_tracked_flows: int,
) -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="maximum tracked flows must be positive",
    ):
        OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=120.0,
            max_tracked_flows=max_tracked_flows,
        )


def test_live_session_starts_in_starting_state() -> None:
    session = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
    )

    assert session.state is OperatorLiveState.STARTING
    assert session.failure is None
    assert not session.state.is_terminal


def test_live_session_follows_normal_lifecycle() -> None:
    starting = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
    )

    running = starting.transition(OperatorLiveState.RUNNING)
    stopping = running.transition(OperatorLiveState.STOPPING)
    completed = stopping.transition(OperatorLiveState.COMPLETED)

    assert starting.state is OperatorLiveState.STARTING
    assert running.state is OperatorLiveState.RUNNING
    assert stopping.state is OperatorLiveState.STOPPING
    assert completed.state is OperatorLiveState.COMPLETED
    assert completed.state.is_terminal


def test_starting_session_can_stop_before_running() -> None:
    session = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
    )

    stopping = session.transition(OperatorLiveState.STOPPING)
    completed = stopping.transition(OperatorLiveState.COMPLETED)

    assert completed.state is OperatorLiveState.COMPLETED


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            OperatorLiveState.STARTING,
            OperatorLiveState.COMPLETED,
        ),
        (
            OperatorLiveState.RUNNING,
            OperatorLiveState.COMPLETED,
        ),
        (
            OperatorLiveState.RUNNING,
            OperatorLiveState.STARTING,
        ),
        (
            OperatorLiveState.STOPPING,
            OperatorLiveState.RUNNING,
        ),
        (
            OperatorLiveState.COMPLETED,
            OperatorLiveState.RUNNING,
        ),
        (
            OperatorLiveState.FAILED,
            OperatorLiveState.RUNNING,
        ),
    ],
)
def test_rejects_invalid_normal_transition(
    source: OperatorLiveState,
    target: OperatorLiveState,
) -> None:
    failure = (
        OperatorLiveFailure(
            code="synthetic_failure",
            message="synthetic failure",
        )
        if source is OperatorLiveState.FAILED
        else None
    )

    session = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
        state=source,
        failure=failure,
    )

    with pytest.raises(
        OperatorLiveSessionError,
        match="invalid live session transition",
    ):
        session.transition(target)


@pytest.mark.parametrize(
    "state",
    [
        OperatorLiveState.STARTING,
        OperatorLiveState.RUNNING,
        OperatorLiveState.STOPPING,
    ],
)
def test_nonterminal_live_session_can_fail(
    state: OperatorLiveState,
) -> None:
    session = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
        state=state,
    )

    failed = session.fail(
        code="capture_error",
        message="capture failed",
    )

    assert failed.state is OperatorLiveState.FAILED
    assert failed.state.is_terminal
    assert failed.failure == OperatorLiveFailure(
        code="capture_error",
        message="capture failed",
    )
    assert failed.failure.as_dict() == {
        "code": "capture_error",
        "message": "capture failed",
    }


@pytest.mark.parametrize(
    "state",
    [
        OperatorLiveState.COMPLETED,
        OperatorLiveState.FAILED,
    ],
)
def test_terminal_live_session_cannot_fail_again(
    state: OperatorLiveState,
) -> None:
    failure = (
        OperatorLiveFailure(
            code="existing_failure",
            message="already failed",
        )
        if state is OperatorLiveState.FAILED
        else None
    )

    session = OperatorLiveSession(
        run_id="live-001",
        configuration=_configuration(),
        state=state,
        failure=failure,
    )

    with pytest.raises(
        OperatorLiveSessionError,
        match="already terminal",
    ):
        session.fail(
            code="new_failure",
            message="new failure",
        )


@pytest.mark.parametrize("run_id", ["", " ", "\t"])
def test_rejects_empty_live_run_id(run_id: str) -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="run ID must not be empty",
    ):
        OperatorLiveSession(
            run_id=run_id,
            configuration=_configuration(),
        )


def test_failed_session_requires_failure_details() -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="must include failure details",
    ):
        OperatorLiveSession(
            run_id="live-001",
            configuration=_configuration(),
            state=OperatorLiveState.FAILED,
        )


def test_nonfailed_session_rejects_failure_details() -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match="must not include failure details",
    ):
        OperatorLiveSession(
            run_id="live-001",
            configuration=_configuration(),
            state=OperatorLiveState.RUNNING,
            failure=OperatorLiveFailure(
                code="unexpected",
                message="unexpected failure",
            ),
        )


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        ("", "failure", "code must not be empty"),
        (" ", "failure", "code must not be empty"),
        ("capture_error", "", "message must not be empty"),
        ("capture_error", " ", "message must not be empty"),
    ],
)
def test_rejects_invalid_live_failure(
    code: str,
    message: str,
    expected: str,
) -> None:
    with pytest.raises(
        OperatorLiveSessionError,
        match=expected,
    ):
        OperatorLiveFailure(
            code=code,
            message=message,
        )
