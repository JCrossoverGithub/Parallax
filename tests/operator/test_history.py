import json
import sqlite3
from pathlib import Path

import pytest

from parallax.operator.history import (
    OperatorHistoryError,
    OperatorHistoryRecord,
    SqliteOperatorHistory,
)
from parallax.replay import ReplayState


def _record(
    run_id: str = "run-001",
    *,
    state: ReplayState = ReplayState.CREATED,
    event_count: int = 0,
) -> OperatorHistoryRecord:
    return OperatorHistoryRecord(
        run_id=run_id,
        source_id="capture.pcap",
        source_sha256="a" * 64,
        state=state,
        time_scale=1.0,
        event_count=event_count,
        failure_code=None,
        failure_message=None,
    )


def test_persists_and_updates_replay_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)

    history.save_replay(_record())

    assert history.get_replay("run-001") == _record()

    completed = _record(
        state=ReplayState.COMPLETED,
        event_count=2,
    )
    history.save_replay(completed)

    assert history.get_replay("run-001") == completed

    # A new instance proves the data is durable across service objects.
    reopened = SqliteOperatorHistory(path)

    assert reopened.get_replay("run-001") == completed


def test_persists_ordered_prediction_events(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")
    history.save_replay(_record())

    history.save_event(
        "run-001",
        1,
        {
            "run_id": "run-001",
            "window": {
                "window_index": 0,
            },
        },
    )
    history.save_event(
        "run-001",
        2,
        {
            "run_id": "run-001",
            "window": {
                "window_index": 1,
            },
        },
    )

    assert history.get_events("run-001") == (
        {
            "run_id": "run-001",
            "window": {
                "window_index": 0,
            },
        },
        {
            "run_id": "run-001",
            "window": {
                "window_index": 1,
            },
        },
    )


def test_lists_replays_newest_first(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    history.save_replay(_record("run-001"))
    history.save_replay(_record("run-002"))
    history.save_replay(_record("run-003"))

    assert [record.run_id for record in history.list_replays(limit=2)] == [
        "run-003",
        "run-002",
    ]


def test_serializes_failure_state() -> None:
    record = OperatorHistoryRecord(
        run_id="run-001",
        source_id="capture.pcap",
        source_sha256="a" * 64,
        state=ReplayState.FAILED,
        time_scale=None,
        event_count=0,
        failure_code="replay_execution_error",
        failure_message="parser failed",
    )

    assert record.as_dict()["failure"] == {
        "code": "replay_execution_error",
        "message": "parser failed",
    }


def test_serializes_absent_failure() -> None:
    assert _record().as_dict()["failure"] is None


def test_missing_replay_returns_none(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    assert history.get_replay("missing") is None
    assert history.get_events("missing") == ()


def test_rejects_invalid_limits_and_event_sequences(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")
    history.save_replay(_record())

    with pytest.raises(
        OperatorHistoryError,
        match="limit must be positive",
    ):
        history.list_replays(limit=0)

    with pytest.raises(
        OperatorHistoryError,
        match="sequence must be positive",
    ):
        history.save_event(
            "run-001",
            0,
            {},
        )


def test_rejects_directory_database_path(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "database"
    directory.mkdir()

    with pytest.raises(
        OperatorHistoryError,
        match="database file",
    ):
        SqliteOperatorHistory(directory)


def test_rejects_invalid_persisted_replay_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)
    history.save_replay(_record())

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE replay_sessions
            SET state = 'nonsense'
            WHERE run_id = 'run-001'
            """
        )

    with pytest.raises(
        OperatorHistoryError,
        match="persisted replay state is invalid",
    ):
        history.get_replay("run-001")


def test_rejects_non_object_event_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)
    history.save_replay(_record())

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_events (
                run_id,
                sequence,
                payload_json
            )
            VALUES (?, ?, ?)
            """,
            (
                "run-001",
                1,
                json.dumps(["not", "an", "object"]),
            ),
        )

    with pytest.raises(
        OperatorHistoryError,
        match="must be a JSON object",
    ):
        history.get_events("run-001")
