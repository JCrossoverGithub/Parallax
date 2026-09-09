from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import cast

import pytest

from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveState,
)
from parallax.operator.service import (
    OperatorLiveConflictError,
    OperatorLiveExecutor,
    OperatorLiveNotFoundError,
    OperatorModelIdentity,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.runtime import RuntimeScorer
from parallax.sensor import (
    CaptureInterface,
    SensorInterfaceError,
)


def _identity() -> OperatorModelIdentity:
    return OperatorModelIdentity(
        model_bundle_sha256="a" * 64,
        calibration_artifact_sha256="b" * 64,
        feature_artifact_sha256="c" * 64,
        split_manifest_sha256="d" * 64,
    )


def _interfaces() -> tuple[CaptureInterface, ...]:
    return (
        CaptureInterface(index=1, name="lo"),
        CaptureInterface(index=2, name="eth0"),
    )


def _resolve_interface(name: str) -> CaptureInterface:
    for interface in _interfaces():
        if interface.name == name:
            return interface

    raise SensorInterfaceError(f"capture interface {name!r} does not exist")


def _service(
    tmp_path: Path,
    *,
    executor: OperatorLiveExecutor | None,
    interface_lister: Callable[[], tuple[CaptureInterface, ...]] = _interfaces,
    interface_resolver: Callable[[str], CaptureInterface] = _resolve_interface,
) -> OperatorReplayService:
    return OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(RuntimeScorer, object()),
        model_identity=_identity(),
        live_executor=executor,
        live_interface_lister=interface_lister,
        live_interface_resolver=interface_resolver,
    )


def _wait_for_state(
    service: OperatorReplayService,
    run_id: str,
    state: OperatorLiveState,
) -> None:
    deadline = monotonic() + 2.0

    while monotonic() < deadline:
        if service.get_live(run_id).state is state:
            return

        sleep(0.01)

    raise AssertionError(f"live session did not reach {state.value!r}")


def test_lists_available_live_interfaces(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
    )

    assert service.list_live_interfaces() == _interfaces()


def test_interface_discovery_error_becomes_service_error(
    tmp_path: Path,
) -> None:
    def fail() -> tuple[CaptureInterface, ...]:
        raise SensorInterfaceError("interface discovery failed")

    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
        interface_lister=fail,
    )

    with pytest.raises(
        OperatorServiceError,
        match="interface discovery failed",
    ):
        service.list_live_interfaces()


def test_rejects_start_when_live_execution_is_not_configured(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=None,
    )

    with pytest.raises(
        OperatorServiceError,
        match="live capture execution is not configured",
    ):
        service.start_live("eth0")


def test_rejects_unknown_live_interface(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
    )

    with pytest.raises(
        OperatorServiceError,
        match="does not exist",
    ):
        service.start_live("missing0")


def test_starts_stops_and_completes_owned_live_session(
    tmp_path: Path,
) -> None:
    entered = Event()

    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
    ) -> None:
        assert configuration.interface == "eth0"
        assert configuration.stale_after_seconds == 120.0
        assert configuration.max_tracked_flows == 4_096

        entered.set()
        assert stop_event.wait(timeout=2.0)

    service = _service(
        tmp_path,
        executor=execute,
    )

    created = service.start_live("eth0")

    assert created.state is OperatorLiveState.STARTING
    assert created.configuration.interface == "eth0"

    assert entered.wait(timeout=2.0)

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.RUNNING,
    )

    stopping = service.stop_live(created.run_id)

    assert stopping.state is OperatorLiveState.STOPPING

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )


def test_only_one_live_session_may_be_active(
    tmp_path: Path,
) -> None:
    entered = Event()

    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
    ) -> None:
        entered.set()
        stop_event.wait(timeout=2.0)

    service = _service(
        tmp_path,
        executor=execute,
    )

    first = service.start_live("eth0")
    assert entered.wait(timeout=2.0)

    with pytest.raises(
        OperatorLiveConflictError,
        match="already active",
    ):
        service.start_live("lo")

    service.stop_live(first.run_id)

    _wait_for_state(
        service,
        first.run_id,
        OperatorLiveState.COMPLETED,
    )


def test_unknown_live_session_is_not_found(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
    )

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="does not exist",
    ):
        service.get_live("missing")

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="does not exist",
    ):
        service.stop_live("missing")


def test_terminal_live_session_rejects_stop(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    with pytest.raises(
        OperatorLiveConflictError,
        match="already terminal",
    ):
        service.stop_live(created.run_id)


def test_normal_executor_return_completes_session(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event: None,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    assert service.get_live(created.run_id).failure is None


def test_executor_failure_becomes_structured_live_failure(
    tmp_path: Path,
) -> None:
    def fail(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
    ) -> None:
        raise RuntimeError("synthetic live failure")

    service = _service(
        tmp_path,
        executor=fail,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.FAILED,
    )

    failed = service.get_live(created.run_id)

    assert failed.failure is not None
    assert failed.failure.code == "live_execution_error"
    assert failed.failure.message == "synthetic live failure"


def test_stop_while_starting_prevents_executor_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_called = Event()

    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
    ) -> None:
        executor_called.set()

    def defer_thread_start(_: object) -> None:
        return None

    monkeypatch.setattr(
        "parallax.operator.service.Thread.start",
        defer_thread_start,
    )

    service = _service(
        tmp_path,
        executor=execute,
    )

    created = service.start_live("eth0")

    assert created.state is OperatorLiveState.STARTING

    stopping = service.stop_live(created.run_id)

    assert stopping.state is OperatorLiveState.STOPPING
    assert not executor_called.is_set()

    service._execute_live(created.run_id)

    completed = service.get_live(created.run_id)

    assert completed.state is OperatorLiveState.COMPLETED
    assert completed.failure is None
    assert not executor_called.is_set()


def test_repeated_stop_request_remains_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
    ) -> None:
        raise AssertionError("executor must not start")

    def defer_thread_start(_: object) -> None:
        return None

    monkeypatch.setattr(
        "parallax.operator.service.Thread.start",
        defer_thread_start,
    )

    service = _service(
        tmp_path,
        executor=execute,
    )

    created = service.start_live("eth0")

    first_stop = service.stop_live(created.run_id)
    second_stop = service.stop_live(created.run_id)

    assert first_stop.state is OperatorLiveState.STOPPING
    assert second_stop.state is OperatorLiveState.STOPPING

    service._execute_live(created.run_id)

    assert service.get_live(created.run_id).state is OperatorLiveState.COMPLETED
