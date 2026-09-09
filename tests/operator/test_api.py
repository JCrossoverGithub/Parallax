import asyncio
from typing import cast

from fastapi.testclient import TestClient

from parallax.operator import create_operator_app
from parallax.operator.api import _stream_events
from parallax.operator.history import (
    OperatorHistoryRecord,
    OperatorLiveHistoryRecord,
)
from parallax.operator.service import (
    OperatorEventBatch,
    OperatorEventCursorError,
    OperatorEventRecord,
    OperatorReplayConflictError,
    OperatorReplayNotFoundError,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.replay import ReplayState


class _Result:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, object]:
        return self._payload


class FakeOperatorService:
    def __init__(self) -> None:
        self.controls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[str, int]] = []

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "active_model": {},
            "sessions": {
                "total": 0,
                "active": 0,
            },
        }

    def start_replay(
        self,
        capture_name: str,
        *,
        time_scale: float | None = 1.0,
    ) -> object:
        if capture_name == "missing.pcap":
            raise OperatorServiceError("capture does not exist")

        return _Result(
            {
                "run_id": "replay-001",
                "source_id": capture_name,
                "state": "created",
                "time_scale": time_scale,
            }
        )

    def get_replay(self, run_id: str) -> object:
        if run_id == "missing":
            raise OperatorReplayNotFoundError("replay does not exist")

        return _Result(
            {
                "run_id": run_id,
                "state": "completed",
            }
        )

    def get_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        if run_id == "missing":
            raise OperatorReplayNotFoundError("replay does not exist")

        return (
            {
                "schema_version": "parallax-runtime-prediction-1",
                "run_id": run_id,
            },
        )

    def pause_replay(self, run_id: str) -> None:
        self._control(run_id, "pause")

    def resume_replay(self, run_id: str) -> None:
        self._control(run_id, "resume")

    def cancel_replay(self, run_id: str) -> None:
        self._control(run_id, "cancel")

    def _control(
        self,
        run_id: str,
        action: str,
    ) -> None:
        if run_id == "missing":
            raise OperatorReplayNotFoundError("replay does not exist")

        if run_id == "terminal":
            raise OperatorReplayConflictError("replay is already terminal")

        self.controls.append((run_id, action))

    def get_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorEventBatch:
        self.batch_calls.append((run_id, after_sequence))

        if run_id == "missing":
            raise OperatorReplayNotFoundError("replay does not exist")

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

        return OperatorEventBatch(
            run_id=run_id,
            events=events,
            last_sequence=1,
            state=ReplayState.COMPLETED,
        )


def _service() -> FakeOperatorService:
    return FakeOperatorService()


def _client(
    service: FakeOperatorService | None = None,
) -> TestClient:
    resolved = _service() if service is None else service

    return TestClient(create_operator_app(cast(OperatorReplayService, resolved)))


def test_health_endpoint() -> None:
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_starts_replay() -> None:
    response = _client().post(
        "/api/v1/replays",
        json={
            "capture": "nonvpn_ssh_capture4.pcap",
            "time_scale": None,
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "run_id": "replay-001",
        "source_id": "nonvpn_ssh_capture4.pcap",
        "state": "created",
        "time_scale": None,
    }


def test_start_replay_maps_service_errors_to_bad_request() -> None:
    response = _client().post(
        "/api/v1/replays",
        json={
            "capture": "missing.pcap",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "capture does not exist",
    }


def test_reads_replay_snapshot() -> None:
    response = _client().get("/api/v1/replays/replay-001")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "replay-001",
        "state": "completed",
    }


def test_unknown_replay_snapshot_returns_not_found() -> None:
    response = _client().get("/api/v1/replays/missing")

    assert response.status_code == 404


def test_reads_replay_events() -> None:
    response = _client().get("/api/v1/replays/replay-001/events")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "replay-001",
        "events": [
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": "replay-001",
            }
        ],
    }


def test_unknown_replay_events_return_not_found() -> None:
    response = _client().get("/api/v1/replays/missing/events")

    assert response.status_code == 404


def test_pause_endpoint_accepts_control_request() -> None:
    service = _service()

    response = _client(service).post("/api/v1/replays/replay-001/pause")

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "replay-001",
        "action": "pause",
        "accepted": True,
    }
    assert service.controls == [("replay-001", "pause")]


def test_resume_endpoint_maps_missing_replay_to_not_found() -> None:
    response = _client().post("/api/v1/replays/missing/resume")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "replay does not exist",
    }


def test_cancel_endpoint_maps_conflict() -> None:
    response = _client().post("/api/v1/replays/terminal/cancel")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "replay is already terminal",
    }


def test_resume_and_cancel_endpoints_accept_requests() -> None:
    service = _service()
    client = _client(service)

    resume = client.post("/api/v1/replays/replay-001/resume")
    cancel = client.post("/api/v1/replays/replay-001/cancel")

    assert resume.status_code == 202
    assert cancel.status_code == 202
    assert service.controls == [
        ("replay-001", "resume"),
        ("replay-001", "cancel"),
    ]


def test_stream_returns_prediction_and_terminal_events() -> None:
    response = _client().get("/api/v1/replays/replay-001/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 1\n" in response.text
    assert "event: prediction\n" in response.text
    assert '"schema_version":"parallax-runtime-prediction-1"' in response.text
    assert "event: replay-terminal\n" in response.text
    assert '"state":"completed"' in response.text


def test_stream_accepts_query_cursor() -> None:
    service = _service()

    response = _client(service).get("/api/v1/replays/replay-001/stream?after=1")

    assert response.status_code == 200
    assert service.batch_calls == [("replay-001", 1)]
    assert "event: prediction\n" not in response.text
    assert "event: replay-terminal\n" in response.text


def test_last_event_id_overrides_query_cursor() -> None:
    service = _service()

    response = _client(service).get(
        "/api/v1/replays/replay-001/stream?after=0",
        headers={
            "Last-Event-ID": "1",
        },
    )

    assert response.status_code == 200
    assert service.batch_calls == [("replay-001", 1)]


def test_invalid_last_event_id_returns_bad_request() -> None:
    response = _client().get(
        "/api/v1/replays/replay-001/stream",
        headers={
            "Last-Event-ID": "not-an-integer",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Last-Event-ID must be an integer",
    }


def test_negative_last_event_id_returns_bad_request() -> None:
    response = _client().get(
        "/api/v1/replays/replay-001/stream",
        headers={
            "Last-Event-ID": "-1",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Last-Event-ID must not be negative",
    }


def test_stream_missing_replay_returns_not_found() -> None:
    response = _client().get("/api/v1/replays/missing/stream")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "replay does not exist",
    }


def test_stream_expired_cursor_returns_conflict() -> None:
    response = _client().get("/api/v1/replays/expired/stream")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "requested events are no longer retained",
    }


class PollThenCompleteService:
    def __init__(self) -> None:
        self.calls = 0

    def get_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorEventBatch:
        self.calls += 1

        return OperatorEventBatch(
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
            state=ReplayState.COMPLETED,
        )


class PollThenCursorErrorService:
    def get_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorEventBatch:
        raise OperatorEventCursorError("requested events are no longer retained")


async def _collect_stream(
    stream: object,
) -> list[str]:
    from collections.abc import AsyncIterator

    resolved = cast(AsyncIterator[str], stream)
    return [chunk async for chunk in resolved]


def test_stream_generator_polls_until_terminal() -> None:
    service = cast(
        OperatorReplayService,
        PollThenCompleteService(),
    )

    initial = OperatorEventBatch(
        run_id="replay-001",
        events=(),
        last_sequence=0,
        state=ReplayState.RUNNING,
    )

    chunks = asyncio.run(
        _collect_stream(
            _stream_events(
                service,
                "replay-001",
                initial,
            )
        )
    )

    assert any("event: prediction" in chunk for chunk in chunks)
    assert any("event: replay-terminal" in chunk for chunk in chunks)


def test_stream_generator_reports_cursor_loss() -> None:
    service = cast(
        OperatorReplayService,
        PollThenCursorErrorService(),
    )

    initial = OperatorEventBatch(
        run_id="replay-001",
        events=(),
        last_sequence=0,
        state=ReplayState.RUNNING,
    )

    chunks = asyncio.run(
        _collect_stream(
            _stream_events(
                service,
                "replay-001",
                initial,
            )
        )
    )

    assert chunks == [
        ('event: stream-error\ndata: {"detail":"requested events are no longer retained"}\n\n')
    ]


def _history_record(run_id: str = "history-001") -> OperatorHistoryRecord:
    from parallax.operator.history import OperatorHistoryRecord
    from parallax.replay import ReplayState

    return OperatorHistoryRecord(
        run_id=run_id,
        source_id="capture.pcap",
        source_sha256="a" * 64,
        state=ReplayState.COMPLETED,
        time_scale=1.0,
        event_count=1,
        failure_code=None,
        failure_message=None,
    )


def _live_history_record(
    run_id: str = "live-history-001",
) -> OperatorLiveHistoryRecord:
    from parallax.operator.live import (
        OperatorLiveState,
    )

    return OperatorLiveHistoryRecord(
        run_id=run_id,
        interface="eth0",
        state=OperatorLiveState.COMPLETED,
        stale_after_seconds=120.0,
        max_tracked_flows=4_096,
        event_count=1,
        failure_code=None,
        failure_message=None,
    )


class FakeHistoryService:
    def list_history(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OperatorHistoryRecord, ...]:
        from parallax.operator.service import OperatorServiceError

        if limit < 1:
            raise OperatorServiceError("history list limit must be positive")

        return (_history_record(),)

    def get_history_replay(
        self,
        run_id: str,
    ) -> OperatorHistoryRecord:
        from parallax.operator.service import (
            OperatorReplayNotFoundError,
        )

        if run_id == "missing":
            raise OperatorReplayNotFoundError("persisted replay does not exist")

        return _history_record(run_id)

    def get_history_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        from parallax.operator.service import (
            OperatorReplayNotFoundError,
        )

        if run_id == "missing":
            raise OperatorReplayNotFoundError("persisted replay does not exist")

        return (
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": run_id,
            },
        )

    def list_live_history(
        self,
        *,
        limit: int = 100,
    ) -> tuple[
        OperatorLiveHistoryRecord,
        ...,
    ]:
        from parallax.operator.service import (
            OperatorServiceError,
        )

        if limit < 1:
            raise OperatorServiceError("live history list limit must be positive")

        return (_live_history_record(),)

    def get_history_live(
        self,
        run_id: str,
    ) -> OperatorLiveHistoryRecord:
        from parallax.operator.service import (
            OperatorLiveNotFoundError,
        )

        if run_id == "missing":
            raise OperatorLiveNotFoundError("persisted live session does not exist")

        return _live_history_record(run_id)

    def get_history_live_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        from parallax.operator.service import (
            OperatorLiveNotFoundError,
        )

        if run_id == "missing":
            raise OperatorLiveNotFoundError("persisted live session does not exist")

        return (
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": run_id,
            },
        )


def _history_client() -> TestClient:
    from typing import cast

    from fastapi.testclient import TestClient

    from parallax.operator.api import create_operator_app
    from parallax.operator.service import OperatorReplayService

    service = cast(
        OperatorReplayService,
        FakeHistoryService(),
    )

    return TestClient(create_operator_app(service))


def test_lists_persisted_replay_history() -> None:
    response = _history_client().get("/api/v1/history?limit=10")

    assert response.status_code == 200

    payload = response.json()

    assert len(payload["replays"]) == 1
    assert payload["replays"][0]["run_id"] == "history-001"
    assert payload["replays"][0]["state"] == "completed"


def test_history_list_rejects_invalid_limit() -> None:
    response = _history_client().get("/api/v1/history?limit=0")

    assert response.status_code == 400
    assert response.json() == {"detail": "history list limit must be positive"}


def test_reads_persisted_replay_summary() -> None:
    response = _history_client().get("/api/v1/history/history-002")

    assert response.status_code == 200
    assert response.json()["run_id"] == "history-002"


def test_missing_persisted_replay_returns_not_found() -> None:
    response = _history_client().get("/api/v1/history/missing")

    assert response.status_code == 404


def test_reads_persisted_prediction_events() -> None:
    response = _history_client().get("/api/v1/history/history-003/events")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "history-003",
        "events": [
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": "history-003",
            }
        ],
    }


def test_missing_persisted_events_return_not_found() -> None:
    response = _history_client().get("/api/v1/history/missing/events")

    assert response.status_code == 404


def test_lists_persisted_live_history() -> None:
    response = _history_client().get("/api/v1/history/live?limit=10")

    assert response.status_code == 200

    payload = response.json()

    assert len(payload["live_sessions"]) == 1

    session = payload["live_sessions"][0]

    assert session["run_id"] == "live-history-001"
    assert session["state"] == "completed"
    assert session["configuration"] == {
        "interface": "eth0",
        "stale_after_seconds": 120.0,
        "max_tracked_flows": 4_096,
    }
    assert session["event_count"] == 1


def test_live_history_list_rejects_invalid_limit() -> None:
    response = _history_client().get("/api/v1/history/live?limit=0")

    assert response.status_code == 400

    assert response.json() == {"detail": ("live history list limit must be positive")}


def test_reads_persisted_live_summary() -> None:
    response = _history_client().get("/api/v1/history/live/live-002")

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "live-002"
    assert payload["state"] == "completed"
    assert payload["configuration"]["interface"] == "eth0"


def test_missing_persisted_live_returns_not_found() -> None:
    response = _history_client().get("/api/v1/history/live/missing")

    assert response.status_code == 404

    assert response.json() == {"detail": ("persisted live session does not exist")}


def test_reads_persisted_live_prediction_events() -> None:
    response = _history_client().get("/api/v1/history/live/live-003/events")

    assert response.status_code == 200

    assert response.json() == {
        "run_id": "live-003",
        "events": [
            {
                "schema_version": ("parallax-runtime-prediction-1"),
                "run_id": "live-003",
            }
        ],
    }


def test_missing_persisted_live_events_return_not_found() -> None:
    response = _history_client().get("/api/v1/history/live/missing/events")

    assert response.status_code == 404

    assert response.json() == {"detail": ("persisted live session does not exist")}
