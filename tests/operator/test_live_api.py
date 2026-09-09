import asyncio
from typing import cast

from fastapi.testclient import TestClient

from parallax.operator.api import (
    _stream_live_events,
    create_operator_app,
)
from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveFailure,
    OperatorLiveSession,
    OperatorLiveState,
)
from parallax.operator.service import (
    OperatorEventCursorError,
    OperatorEventRecord,
    OperatorLiveConflictError,
    OperatorLiveEventBatch,
    OperatorLiveNotFoundError,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.sensor import CaptureInterface


def _session(
    run_id: str,
    *,
    state: OperatorLiveState,
) -> OperatorLiveSession:
    failure = None

    if state is OperatorLiveState.FAILED:
        failure = OperatorLiveFailure(
            code="live_execution_error",
            message="capture failed",
        )

    return OperatorLiveSession(
        run_id=run_id,
        configuration=OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
        ),
        state=state,
        failure=failure,
    )


class FakeLiveService:
    def __init__(self) -> None:
        self.live_batch_calls: list[tuple[str, int]] = []
        self.stop_calls: list[str] = []

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "active_model": {},
            "sessions": {
                "total": 0,
                "active": 0,
            },
        }

    def list_live_interfaces(
        self,
    ) -> tuple[CaptureInterface, ...]:
        return (
            CaptureInterface(index=1, name="lo"),
            CaptureInterface(index=2, name="eth0"),
        )

    def start_live(
        self,
        interface: str,
    ) -> OperatorLiveSession:
        if interface == "missing0":
            raise OperatorServiceError("capture interface 'missing0' does not exist")

        if interface == "busy0":
            raise OperatorLiveConflictError("a live session is already active")

        return _session(
            "live-001",
            state=OperatorLiveState.STARTING,
        )

    def get_live(
        self,
        run_id: str,
    ) -> OperatorLiveSession:
        if run_id == "missing":
            raise OperatorLiveNotFoundError("live session does not exist")

        if run_id == "failed":
            return _session(
                run_id,
                state=OperatorLiveState.FAILED,
            )

        return _session(
            run_id,
            state=OperatorLiveState.COMPLETED,
        )

    def get_live_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        if run_id == "missing":
            raise OperatorLiveNotFoundError("live session does not exist")

        return (
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": run_id,
            },
        )

    def stop_live(
        self,
        run_id: str,
    ) -> OperatorLiveSession:
        if run_id == "missing":
            raise OperatorLiveNotFoundError("live session does not exist")

        if run_id == "terminal":
            raise OperatorLiveConflictError("live session is already terminal")

        self.stop_calls.append(run_id)

        return _session(
            run_id,
            state=OperatorLiveState.STOPPING,
        )

    def get_live_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorLiveEventBatch:
        self.live_batch_calls.append((run_id, after_sequence))

        if run_id == "missing":
            raise OperatorLiveNotFoundError("live session does not exist")

        if run_id == "expired":
            raise OperatorEventCursorError("requested events are no longer retained")

        events: tuple[OperatorEventRecord, ...]

        if after_sequence >= 1:
            events = ()
        else:
            events = (
                OperatorEventRecord(
                    sequence=1,
                    payload={
                        "schema_version": ("parallax-runtime-prediction-1"),
                        "run_id": run_id,
                    },
                ),
            )

        return OperatorLiveEventBatch(
            run_id=run_id,
            events=events,
            last_sequence=1,
            state=OperatorLiveState.COMPLETED,
        )


class FailingInterfaceService(FakeLiveService):
    def list_live_interfaces(
        self,
    ) -> tuple[CaptureInterface, ...]:
        raise OperatorServiceError("interface discovery failed")


def _client(
    service: FakeLiveService | None = None,
) -> TestClient:
    resolved = FakeLiveService() if service is None else service

    return TestClient(
        create_operator_app(
            cast(
                OperatorReplayService,
                resolved,
            )
        )
    )


def test_lists_live_interfaces() -> None:
    response = _client().get("/api/v1/live/interfaces")

    assert response.status_code == 200
    assert response.json() == {
        "interfaces": [
            {
                "index": 1,
                "name": "lo",
            },
            {
                "index": 2,
                "name": "eth0",
            },
        ]
    }


def test_live_interface_discovery_error_is_bad_request() -> None:
    response = _client(FailingInterfaceService()).get("/api/v1/live/interfaces")

    assert response.status_code == 400
    assert response.json() == {"detail": "interface discovery failed"}


def test_starts_live_session() -> None:
    response = _client().post(
        "/api/v1/live",
        json={
            "interface": "eth0",
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "run_id": "live-001",
        "state": "starting",
        "configuration": {
            "interface": "eth0",
            "stale_after_seconds": 120.0,
            "max_tracked_flows": 4_096,
        },
        "failure": None,
    }


def test_start_live_maps_service_error() -> None:
    response = _client().post(
        "/api/v1/live",
        json={
            "interface": "missing0",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": ("capture interface 'missing0' does not exist")}


def test_start_live_maps_active_session_conflict() -> None:
    response = _client().post(
        "/api/v1/live",
        json={
            "interface": "busy0",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "a live session is already active"}


def test_reads_live_session() -> None:
    response = _client().get("/api/v1/live/live-001")

    assert response.status_code == 200
    assert response.json()["run_id"] == "live-001"
    assert response.json()["state"] == "completed"
    assert response.json()["failure"] is None


def test_failed_live_session_serializes_failure() -> None:
    response = _client().get("/api/v1/live/failed")

    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert response.json()["failure"] == {
        "code": "live_execution_error",
        "message": "capture failed",
    }


def test_missing_live_session_returns_not_found() -> None:
    response = _client().get("/api/v1/live/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "live session does not exist"}


def test_reads_live_prediction_events() -> None:
    response = _client().get("/api/v1/live/live-001/events")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "live-001",
        "events": [
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": "live-001",
            }
        ],
    }


def test_missing_live_events_return_not_found() -> None:
    response = _client().get("/api/v1/live/missing/events")

    assert response.status_code == 404


def test_stop_live_session() -> None:
    service = FakeLiveService()

    response = _client(service).post("/api/v1/live/live-001/stop")

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "live-001",
        "action": "stop",
        "accepted": True,
        "state": "stopping",
    }
    assert service.stop_calls == ["live-001"]


def test_stop_missing_live_session_returns_not_found() -> None:
    response = _client().post("/api/v1/live/missing/stop")

    assert response.status_code == 404


def test_stop_terminal_live_session_returns_conflict() -> None:
    response = _client().post("/api/v1/live/terminal/stop")

    assert response.status_code == 409
    assert response.json() == {"detail": "live session is already terminal"}


def test_live_stream_returns_prediction_and_terminal() -> None:
    response = _client().get("/api/v1/live/live-001/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    assert "id: 1\n" in response.text
    assert "event: prediction\n" in response.text
    assert "event: live-terminal\n" in response.text
    assert '"state":"completed"' in response.text


def test_live_stream_accepts_query_cursor() -> None:
    service = FakeLiveService()

    response = _client(service).get("/api/v1/live/live-001/stream?after=1")

    assert response.status_code == 200
    assert service.live_batch_calls == [("live-001", 1)]
    assert "event: prediction\n" not in response.text
    assert "event: live-terminal\n" in response.text


def test_live_stream_accepts_last_event_id() -> None:
    service = FakeLiveService()

    response = _client(service).get(
        "/api/v1/live/live-001/stream?after=0",
        headers={
            "Last-Event-ID": "1",
        },
    )

    assert response.status_code == 200
    assert service.live_batch_calls == [("live-001", 1)]


def test_live_stream_missing_session_returns_not_found() -> None:
    response = _client().get("/api/v1/live/missing/stream")

    assert response.status_code == 404


def test_live_stream_expired_cursor_returns_conflict() -> None:
    response = _client().get("/api/v1/live/expired/stream")

    assert response.status_code == 409
    assert response.json() == {"detail": ("requested events are no longer retained")}


class PollThenCompleteLiveService:
    def get_live_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorLiveEventBatch:
        return OperatorLiveEventBatch(
            run_id=run_id,
            events=(
                OperatorEventRecord(
                    sequence=1,
                    payload={
                        "run_id": run_id,
                    },
                ),
            ),
            last_sequence=1,
            state=OperatorLiveState.COMPLETED,
        )


class PollThenLiveCursorErrorService:
    def get_live_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorLiveEventBatch:
        raise OperatorEventCursorError("requested events are no longer retained")


async def _collect_stream(
    stream: object,
) -> list[str]:
    from collections.abc import AsyncIterator

    resolved = cast(
        AsyncIterator[str],
        stream,
    )

    return [chunk async for chunk in resolved]


def test_live_stream_generator_polls_until_terminal() -> None:
    service = cast(
        OperatorReplayService,
        PollThenCompleteLiveService(),
    )

    initial = OperatorLiveEventBatch(
        run_id="live-001",
        events=(),
        last_sequence=0,
        state=OperatorLiveState.RUNNING,
    )

    chunks = asyncio.run(
        _collect_stream(
            _stream_live_events(
                service,
                "live-001",
                initial,
            )
        )
    )

    assert any("event: prediction" in chunk for chunk in chunks)
    assert any("event: live-terminal" in chunk for chunk in chunks)


def test_live_stream_generator_reports_cursor_loss() -> None:
    service = cast(
        OperatorReplayService,
        PollThenLiveCursorErrorService(),
    )

    initial = OperatorLiveEventBatch(
        run_id="live-001",
        events=(),
        last_sequence=0,
        state=OperatorLiveState.RUNNING,
    )

    chunks = asyncio.run(
        _collect_stream(
            _stream_live_events(
                service,
                "live-001",
                initial,
            )
        )
    )

    assert chunks == [
        ('event: stream-error\ndata: {"detail":"requested events are no longer retained"}\n\n')
    ]
