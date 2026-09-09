from typing import cast

from fastapi.testclient import TestClient

from parallax.operator import create_operator_app
from parallax.operator.service import OperatorReplayService


class FakeOperatorService:
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
            from parallax.operator.service import OperatorServiceError

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
            from parallax.operator.service import OperatorReplayNotFoundError

            raise OperatorReplayNotFoundError("replay does not exist")

        return _Result(
            {
                "run_id": run_id,
                "state": "completed",
            }
        )

    def get_events(self, run_id: str) -> tuple[dict[str, object], ...]:
        if run_id == "missing":
            from parallax.operator.service import OperatorReplayNotFoundError

            raise OperatorReplayNotFoundError("replay does not exist")

        return (
            {
                "schema_version": "parallax-runtime-prediction-1",
                "run_id": run_id,
            },
        )


class _Result:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, object]:
        return self._payload


def _client() -> TestClient:
    service = cast(OperatorReplayService, FakeOperatorService())
    return TestClient(create_operator_app(service))


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
                "schema_version": "parallax-runtime-prediction-1",
                "run_id": "replay-001",
            }
        ],
    }


def test_unknown_replay_events_return_not_found() -> None:
    response = _client().get("/api/v1/replays/missing/events")

    assert response.status_code == 404
