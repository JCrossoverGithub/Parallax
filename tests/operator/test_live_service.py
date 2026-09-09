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


def test_active_live_session_is_discoverable(
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

    assert service.get_active_live() is None

    created = service.start_live("eth0")

    assert entered.wait(timeout=2.0)

    active = service.get_active_live()

    assert active is not None
    assert active.run_id == created.run_id
    assert not active.state.is_terminal

    service.stop_live(created.run_id)

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    assert service.get_active_live() is None


def test_live_event_snapshot_preserves_global_cursor(
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
                start_offset_seconds=float(index),
                end_offset_seconds=float(index + 1),
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

    snapshot = service.get_live_event_snapshot(
        created.run_id,
    )

    assert snapshot.run_id == created.run_id
    assert len(snapshot.events) == 2
    assert snapshot.last_sequence == 3
    assert snapshot.state is OperatorLiveState.COMPLETED

    window_indices: list[object] = []

    for event in snapshot.events:
        window = event["window"]
        assert isinstance(window, dict)
        window_indices.append(window["window_index"])

    assert window_indices == [1, 2]


def test_persists_completed_live_session_and_events(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        SqliteOperatorHistory,
    )
    from parallax.runtime import (
        RuntimePredictionEvent,
        RuntimeScorer,
    )

    class FakeEvent:
        def __init__(
            self,
            run_id: str,
        ) -> None:
            self.run_id = run_id

        def as_dict(
            self,
        ) -> dict[str, object]:
            return {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": self.run_id,
                "window": {
                    "capture_id": (f"live:eth0:{self.run_id}"),
                    "window_index": 0,
                },
            }

    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: Callable[
            [RuntimePredictionEvent],
            None,
        ],
    ) -> None:
        assert configuration.interface == "eth0"
        assert not stop_event.is_set()

        handle_event(
            cast(
                RuntimePredictionEvent,
                FakeEvent(run_id),
            )
        )

    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
        live_executor=execute,
        live_interface_lister=_interfaces,
        live_interface_resolver=_resolve_interface,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.COMPLETED,
    )

    persisted = service.get_history_live(created.run_id)
    events = service.get_history_live_events(created.run_id)

    assert persisted.state is OperatorLiveState.COMPLETED
    assert persisted.interface == "eth0"
    assert persisted.stale_after_seconds == 120.0
    assert persisted.max_tracked_flows == 4_096
    assert persisted.event_count == 1
    assert persisted.failure_code is None

    assert len(events) == 1
    assert events[0]["run_id"] == created.run_id

    reopened = SqliteOperatorHistory(path)

    restarted = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=reopened,
    )

    assert [record.run_id for record in restarted.list_live_history()] == [created.run_id]

    assert restarted.get_history_live(created.run_id) == persisted

    assert restarted.get_history_live_events(created.run_id) == events


def test_persists_failed_live_session(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        SqliteOperatorHistory,
    )
    from parallax.runtime import RuntimeScorer

    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        raise RuntimeError("synthetic persisted failure")

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
        live_executor=execute,
        live_interface_lister=_interfaces,
        live_interface_resolver=_resolve_interface,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.FAILED,
    )

    persisted = service.get_history_live(created.run_id)

    assert persisted.state is OperatorLiveState.FAILED
    assert persisted.failure_code == "live_execution_error"
    assert persisted.failure_message == "synthetic persisted failure"


def test_service_marks_interrupted_live_history_failed(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        OperatorLiveHistoryRecord,
        SqliteOperatorHistory,
    )
    from parallax.runtime import RuntimeScorer

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    history.save_live_session(
        OperatorLiveHistoryRecord(
            run_id="interrupted-live",
            interface="eth0",
            state=OperatorLiveState.RUNNING,
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
            event_count=7,
            failure_code=None,
            failure_message=None,
        )
    )

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
    )

    persisted = service.get_history_live("interrupted-live")

    assert persisted.state is OperatorLiveState.FAILED
    assert persisted.event_count == 7
    assert persisted.failure_code == "operator_restart"


def test_service_without_history_has_no_live_history(
    tmp_path: Path,
) -> None:
    def execute(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        return None

    service = _service(
        tmp_path,
        executor=execute,
    )

    assert service.list_live_history() == ()

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="persisted live session",
    ):
        service.get_history_live("missing")

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="persisted live session",
    ):
        service.get_history_live_events("missing")


def test_live_history_validation_errors_become_service_errors(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        SqliteOperatorHistory,
    )
    from parallax.runtime import RuntimeScorer

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
    )

    with pytest.raises(
        OperatorServiceError,
        match=("live history list limit must be positive"),
    ):
        service.list_live_history(limit=0)


def test_configured_history_rejects_unknown_live_session(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        SqliteOperatorHistory,
    )
    from parallax.runtime import RuntimeScorer

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
    )

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="persisted live session",
    ):
        service.get_history_live("missing")

    with pytest.raises(
        OperatorLiveNotFoundError,
        match="persisted live session",
    ):
        service.get_history_live_events("missing")


def test_structured_executor_failure_preserves_code(
    tmp_path: Path,
) -> None:
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )

    def fail(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        raise OperatorLiveExecutionError(
            "capture permission denied",
            code="capture_error",
        )

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
    assert failed.failure.code == "capture_error"
    assert failed.failure.message == "capture permission denied"


def test_structured_executor_failure_is_persisted(
    tmp_path: Path,
) -> None:
    from typing import cast

    from parallax.operator.history import (
        SqliteOperatorHistory,
    )
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )
    from parallax.runtime import RuntimeScorer

    def fail(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        raise OperatorLiveExecutionError(
            "sensor capture failed",
            code="capture_error",
        )

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=cast(
            RuntimeScorer,
            object(),
        ),
        model_identity=_identity(),
        history=history,
        live_executor=fail,
        live_interface_lister=_interfaces,
        live_interface_resolver=_resolve_interface,
    )

    created = service.start_live("eth0")

    _wait_for_state(
        service,
        created.run_id,
        OperatorLiveState.FAILED,
    )

    persisted = service.get_history_live(created.run_id)

    assert persisted.failure_code == "capture_error"
    assert persisted.failure_message == "sensor capture failed"


def test_flow_capacity_failure_is_operator_visible(
    tmp_path: Path,
) -> None:
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )

    def fail(
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: object,
    ) -> None:
        raise OperatorLiveExecutionError(
            (f"runtime flow capacity reached: {configuration.max_tracked_flows}"),
            code="flow_capacity_exceeded",
        )

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
    assert failed.failure.code == "flow_capacity_exceeded"
    assert failed.failure.message == "runtime flow capacity reached: 4096"
