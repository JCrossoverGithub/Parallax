from math import inf, nan
from uuid import UUID

import pytest

from parallax.replay import (
    ReplayConfiguration,
    ReplayDomainError,
    ReplayFailure,
    ReplaySession,
    ReplaySessionId,
    ReplayState,
    allowed_replay_transitions,
)

SESSION_ID = "4d524cee-1288-45d7-9c6f-fdbab196bb26"
SOURCE_SHA256 = "a" * 64


def _session(*, state: ReplayState = ReplayState.CREATED) -> ReplaySession:
    failure = ReplayFailure(code="fixture_failure", message="fixture failed")
    return ReplaySession(
        session_id=ReplaySessionId.parse(SESSION_ID),
        source_id="synthetic-replay.pcap",
        source_sha256=SOURCE_SHA256,
        configuration=ReplayConfiguration(),
        state=state,
        failure=failure if state is ReplayState.FAILED else None,
    )


def test_replay_session_id_parses_and_formats_canonical_uuid() -> None:
    session_id = ReplaySessionId.parse(SESSION_ID.upper())

    assert session_id.value == UUID(SESSION_ID)
    assert str(session_id) == SESSION_ID


def test_replay_session_id_can_generate_new_identity() -> None:
    first = ReplaySessionId.new()
    second = ReplaySessionId.new()

    assert isinstance(first.value, UUID)
    assert isinstance(second.value, UUID)
    assert first != second


def test_replay_session_id_rejects_invalid_text() -> None:
    with pytest.raises(ReplayDomainError, match="valid UUID"):
        ReplaySessionId.parse("not-a-uuid")


def test_default_replay_configuration_is_one_times_real_time() -> None:
    configuration = ReplayConfiguration()

    assert configuration.time_scale == 1.0
    assert configuration.maximum_speed is False


def test_none_time_scale_represents_maximum_speed() -> None:
    configuration = ReplayConfiguration(time_scale=None)

    assert configuration.maximum_speed is True


@pytest.mark.parametrize("time_scale", [0.0, -1.0, inf, -inf, nan])
def test_replay_configuration_rejects_invalid_time_scale(time_scale: float) -> None:
    with pytest.raises(ReplayDomainError, match="finite and greater than zero"):
        ReplayConfiguration(time_scale=time_scale)


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        ("", "failure", "code"),
        ("   ", "failure", "code"),
        ("failure", "", "message"),
        ("failure", "   ", "message"),
    ],
)
def test_replay_failure_requires_code_and_message(
    code: str,
    message: str,
    expected: str,
) -> None:
    with pytest.raises(ReplayDomainError, match=expected):
        ReplayFailure(code=code, message=message)


def test_created_session_retains_source_identity_and_configuration() -> None:
    session = _session()

    assert str(session.session_id) == SESSION_ID
    assert session.source_id == "synthetic-replay.pcap"
    assert session.source_sha256 == SOURCE_SHA256
    assert session.configuration == ReplayConfiguration()
    assert session.state is ReplayState.CREATED
    assert session.failure is None
    assert session.is_terminal is False


@pytest.mark.parametrize("source_id", ["", "   "])
def test_replay_session_rejects_empty_source_identity(source_id: str) -> None:
    with pytest.raises(ReplayDomainError, match="source ID"):
        ReplaySession(
            session_id=ReplaySessionId.parse(SESSION_ID),
            source_id=source_id,
            source_sha256=SOURCE_SHA256,
            configuration=ReplayConfiguration(),
        )


@pytest.mark.parametrize(
    "source_sha256",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_replay_session_rejects_invalid_source_sha256(source_sha256: str) -> None:
    with pytest.raises(ReplayDomainError, match="64 lowercase hexadecimal digits"):
        ReplaySession(
            session_id=ReplaySessionId.parse(SESSION_ID),
            source_id="synthetic-replay.pcap",
            source_sha256=source_sha256,
            configuration=ReplayConfiguration(),
        )


def test_failed_session_requires_failure_details() -> None:
    with pytest.raises(ReplayDomainError, match="must retain a failure"):
        ReplaySession(
            session_id=ReplaySessionId.parse(SESSION_ID),
            source_id="synthetic-replay.pcap",
            source_sha256=SOURCE_SHA256,
            configuration=ReplayConfiguration(),
            state=ReplayState.FAILED,
        )


def test_nonfailed_session_rejects_failure_details() -> None:
    with pytest.raises(ReplayDomainError, match="only failed"):
        ReplaySession(
            session_id=ReplaySessionId.parse(SESSION_ID),
            source_id="synthetic-replay.pcap",
            source_sha256=SOURCE_SHA256,
            configuration=ReplayConfiguration(),
            state=ReplayState.RUNNING,
            failure=ReplayFailure(code="unexpected", message="not terminal"),
        )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (ReplayState.CREATED, ReplayState.RUNNING),
        (ReplayState.CREATED, ReplayState.CANCELLED),
        (ReplayState.RUNNING, ReplayState.PAUSED),
        (ReplayState.RUNNING, ReplayState.COMPLETED),
        (ReplayState.RUNNING, ReplayState.CANCELLED),
        (ReplayState.PAUSED, ReplayState.RUNNING),
        (ReplayState.PAUSED, ReplayState.CANCELLED),
    ],
)
def test_nonfailure_lifecycle_transitions_are_immutable(
    source: ReplayState,
    target: ReplayState,
) -> None:
    original = _session(state=source)

    transitioned = original.transition(target)

    assert original.state is source
    assert transitioned.state is target
    assert transitioned.failure is None


@pytest.mark.parametrize(
    "source",
    [
        ReplayState.CREATED,
        ReplayState.RUNNING,
        ReplayState.PAUSED,
    ],
)
def test_active_states_can_transition_to_failed(source: ReplayState) -> None:
    original = _session(state=source)
    failure = ReplayFailure(code="pcap_read_error", message="capture could not be read")

    transitioned = original.transition(ReplayState.FAILED, failure=failure)

    assert transitioned.state is ReplayState.FAILED
    assert transitioned.failure == failure
    assert transitioned.is_terminal is True


def test_transition_to_failed_requires_failure_details() -> None:
    with pytest.raises(ReplayDomainError, match="requires a replay failure"):
        _session().transition(ReplayState.FAILED)


def test_nonfailed_transition_rejects_failure_details() -> None:
    failure = ReplayFailure(code="unexpected", message="should not be retained")

    with pytest.raises(ReplayDomainError, match="only valid for transition to failed"):
        _session().transition(ReplayState.RUNNING, failure=failure)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (ReplayState.CREATED, ReplayState.PAUSED),
        (ReplayState.CREATED, ReplayState.COMPLETED),
        (ReplayState.RUNNING, ReplayState.CREATED),
        (ReplayState.PAUSED, ReplayState.COMPLETED),
        (ReplayState.PAUSED, ReplayState.PAUSED),
    ],
)
def test_invalid_nonterminal_transitions_are_rejected(
    source: ReplayState,
    target: ReplayState,
) -> None:
    with pytest.raises(ReplayDomainError, match="invalid replay transition"):
        _session(state=source).transition(target)


@pytest.mark.parametrize(
    "state",
    [
        ReplayState.COMPLETED,
        ReplayState.FAILED,
        ReplayState.CANCELLED,
    ],
)
def test_terminal_states_have_no_allowed_transitions(state: ReplayState) -> None:
    session = _session(state=state)

    assert session.is_terminal is True
    assert allowed_replay_transitions(state) == frozenset()

    for target in ReplayState:
        with pytest.raises(ReplayDomainError, match="invalid replay transition"):
            session.transition(target)


def test_allowed_transitions_are_exposed_as_immutable_sets() -> None:
    assert allowed_replay_transitions(ReplayState.CREATED) == frozenset(
        {
            ReplayState.RUNNING,
            ReplayState.FAILED,
            ReplayState.CANCELLED,
        }
    )
