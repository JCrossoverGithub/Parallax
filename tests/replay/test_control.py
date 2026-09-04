import pytest

from parallax.replay.control import (
    ReplayControl,
    ReplayControlError,
    ReplayControlState,
)
from parallax.replay.domain import ReplayDomainError


def _assert_state(control: ReplayControl, expected: ReplayControlState) -> None:
    assert control.state is expected


def test_replay_control_starts_running() -> None:
    control = ReplayControl()

    _assert_state(control, ReplayControlState.RUNNING)
    assert control.is_paused is False
    assert control.is_cancelled is False


def test_pause_and_resume_change_requested_state() -> None:
    control = ReplayControl()

    control.pause()

    _assert_state(control, ReplayControlState.PAUSED)
    assert control.is_paused is True
    assert control.is_cancelled is False

    control.resume()

    _assert_state(control, ReplayControlState.RUNNING)
    assert control.is_paused is False


def test_repeated_pause_and_resume_requests_are_idempotent() -> None:
    control = ReplayControl()

    control.pause()
    control.pause()
    _assert_state(control, ReplayControlState.PAUSED)

    control.resume()
    control.resume()
    _assert_state(control, ReplayControlState.RUNNING)


def test_cancel_is_terminal_and_idempotent() -> None:
    control = ReplayControl()

    control.cancel()
    control.cancel()

    _assert_state(control, ReplayControlState.CANCELLED)
    assert control.is_cancelled is True
    assert control.is_paused is False


def test_cancelled_control_rejects_pause_and_resume() -> None:
    control = ReplayControl()
    control.cancel()

    with pytest.raises(ReplayControlError, match="cannot be paused"):
        control.pause()

    with pytest.raises(ReplayControlError, match="cannot be resumed"):
        control.resume()


def test_control_errors_are_replay_domain_errors() -> None:
    control = ReplayControl()
    control.cancel()

    with pytest.raises(ReplayDomainError):
        control.resume()
