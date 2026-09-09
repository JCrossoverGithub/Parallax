from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import cast

import pytest

from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveState,
)
from parallax.operator.service import (
    OperatorEventCursorError,
    OperatorLiveConflictError,
    OperatorLiveExecutor,
    OperatorLiveNotFoundError,
    OperatorModelIdentity,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.runtime import (
    RuntimePredictionEvent,
    RuntimeScorer,
)
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
    event_history_limit: int = 10_000,
) -> OperatorReplayService:
    return OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(RuntimeScorer, object()),
        model_identity=_identity(),
        live_executor=executor,
        event_history_limit=event_history_limit,
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
        executor=lambda run_id, configuration, stop_event, handle_event: None,
    )

    assert service.list_live_interfaces() == _interfaces()


def test_interface_discovery_error_becomes_service_error(
    tmp_path: Path,
) -> None:
    def fail() -> tuple[CaptureInterface, ...]:
        raise SensorInterfaceError("interface discovery failed")

    service = _service(
        tmp_path,
        executor=lambda run_id, configuration, stop_event, handle_event: None,
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
        executor=lambda run_id, configuration, stop_event, handle_event: None,
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
        handle_event: object,
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
        handle_event: object,
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
        executor=lambda run_id, configuration, stop_event, handle_event: None,
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
        executor=lambda run_id, configuration, stop_event, handle_event: None,
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
        executor=lambda run_id, configuration, stop_event, handle_event: None,
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
        handle_event: object,
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
        handle_event: object,
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
        handle_event: object,
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


def test_live_session_retains_prediction_events(
    tmp_path: Path,
) -> None:
    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        callback = cast(
            Callable[[RuntimePredictionEvent], None],
            handle_event,
        )

        prediction = PrototypeRuntimePrediction(
            window_id="window-001",
            capture_id=f"live:eth0:{run_id}",
            flow_id="flow-001",
            window_index=0,
            start_offset_seconds=0.0,
            end_offset_seconds=10.0,
            packet_count=21,
            category_order=CATEGORY_LABELS,
            class_probabilities=(
                0.7,
                0.1,
                0.1,
                0.05,
                0.05,
            ),
            predicted_class_index=0,
            predicted_category=CATEGORY_LABELS[0],
            raw_confidence=0.7,
            relative_mahalanobis_distance=1.0,
            ood_score=0.1,
            model_bundle_sha256="a" * 64,
            calibration_artifact_sha256="b" * 64,
            feature_artifact_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
        )

        callback(
            RuntimePredictionEvent(
                run_id=run_id,
                prediction=prediction,
            )
        )

    service = _service(
        tmp_path,
        executor=execute,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    events = service.get_live_events(created.run_id)

    assert len(events) == 1
    assert events[0]["run_id"] == created.run_id


def test_live_event_batch_sequences_and_validates_cursor(
    tmp_path: Path,
) -> None:
    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        callback = cast(
            Callable[[RuntimePredictionEvent], None],
            handle_event,
        )

        for index in range(3):
            prediction = PrototypeRuntimePrediction(
                window_id=f"window-{index}",
                capture_id=f"live:eth0:{run_id}",
                flow_id="flow-001",
                window_index=index,
                start_offset_seconds=float(index * 10),
                end_offset_seconds=float((index + 1) * 10),
                packet_count=21,
                category_order=CATEGORY_LABELS,
                class_probabilities=(
                    0.7,
                    0.1,
                    0.1,
                    0.05,
                    0.05,
                ),
                predicted_class_index=0,
                predicted_category=CATEGORY_LABELS[0],
                raw_confidence=0.7,
                relative_mahalanobis_distance=1.0,
                ood_score=0.1,
                model_bundle_sha256="a" * 64,
                calibration_artifact_sha256="b" * 64,
                feature_artifact_sha256="c" * 64,
                split_manifest_sha256="d" * 64,
            )

            callback(
                RuntimePredictionEvent(
                    run_id=run_id,
                    prediction=prediction,
                )
            )

    service = _service(
        tmp_path,
        executor=execute,
        event_history_limit=2,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    batch = service.get_live_event_batch(
        created.run_id,
        after_sequence=1,
    )

    assert [event.sequence for event in batch.events] == [2, 3]
    assert batch.last_sequence == 3
    assert batch.state is OperatorLiveState.COMPLETED

    with pytest.raises(
        OperatorEventCursorError,
        match="no longer retained",
    ):
        service.get_live_event_batch(
            created.run_id,
            after_sequence=0,
        )

    with pytest.raises(
        OperatorEventCursorError,
        match="ahead of the live session",
    ):
        service.get_live_event_batch(
            created.run_id,
            after_sequence=4,
        )

    with pytest.raises(
        OperatorEventCursorError,
        match="must not be negative",
    ):
        service.get_live_event_batch(
            created.run_id,
            after_sequence=-1,
        )
