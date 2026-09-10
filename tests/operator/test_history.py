import json
import sqlite3
from pathlib import Path

import pytest

from parallax.operator.history import (
    OperatorHistoryError,
    OperatorHistoryRecord,
    OperatorLiveHistoryRecord,
    SqliteOperatorHistory,
)
from parallax.operator.live import OperatorLiveState
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


def _live_record(
    run_id: str = "live-001",
    *,
    state: OperatorLiveState = (OperatorLiveState.STARTING),
    event_count: int = 0,
) -> OperatorLiveHistoryRecord:
    return OperatorLiveHistoryRecord(
        run_id=run_id,
        interface="eth0",
        state=state,
        stale_after_seconds=120.0,
        max_tracked_flows=4_096,
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


def test_persists_and_updates_live_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)

    history.save_live_session(_live_record())

    assert history.get_live_session("live-001") == _live_record()

    completed = _live_record(
        state=OperatorLiveState.COMPLETED,
        event_count=2,
    )
    history.save_live_session(completed)

    assert history.get_live_session("live-001") == completed

    reopened = SqliteOperatorHistory(path)

    assert reopened.get_live_session("live-001") == completed


def test_persists_ordered_live_prediction_events(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")
    history.save_live_session(_live_record())

    history.save_live_event(
        "live-001",
        1,
        {
            "run_id": "live-001",
            "window": {
                "window_index": 0,
            },
        },
    )
    history.save_live_event(
        "live-001",
        2,
        {
            "run_id": "live-001",
            "window": {
                "window_index": 1,
            },
        },
    )

    assert history.get_live_events("live-001") == (
        {
            "run_id": "live-001",
            "window": {
                "window_index": 0,
            },
        },
        {
            "run_id": "live-001",
            "window": {
                "window_index": 1,
            },
        },
    )


def test_lists_live_sessions_newest_first(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    history.save_live_session(_live_record("live-001"))
    history.save_live_session(_live_record("live-002"))
    history.save_live_session(_live_record("live-003"))

    assert [record.run_id for record in history.list_live_sessions(limit=2)] == [
        "live-003",
        "live-002",
    ]


def test_serializes_live_history_failure() -> None:
    record = OperatorLiveHistoryRecord(
        run_id="live-001",
        interface="eth0",
        state=OperatorLiveState.FAILED,
        stale_after_seconds=120.0,
        max_tracked_flows=4_096,
        event_count=3,
        failure_code="capture_error",
        failure_message="permission denied",
    )

    assert record.as_dict() == {
        "run_id": "live-001",
        "state": "failed",
        "configuration": {
            "interface": "eth0",
            "stale_after_seconds": 120.0,
            "max_tracked_flows": 4_096,
        },
        "event_count": 3,
        "failure": {
            "code": "capture_error",
            "message": "permission denied",
        },
    }


def test_serializes_live_history_without_failure() -> None:
    assert _live_record().as_dict()["failure"] is None


def test_missing_live_history_returns_empty_results(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    assert history.get_live_session("missing") is None

    assert history.get_live_events("missing") == ()


def test_rejects_invalid_live_history_limits_and_sequences(
    tmp_path: Path,
) -> None:
    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")
    history.save_live_session(_live_record())

    with pytest.raises(
        OperatorHistoryError,
        match="live history list limit must be positive",
    ):
        history.list_live_sessions(limit=0)

    with pytest.raises(
        OperatorHistoryError,
        match="live event sequence must be positive",
    ):
        history.save_live_event(
            "live-001",
            0,
            {},
        )


def test_rejects_invalid_persisted_live_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    history = SqliteOperatorHistory(path)
    history.save_live_session(_live_record())

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE live_sessions
            SET state = 'nonsense'
            WHERE run_id = 'live-001'
            """
        )

    with pytest.raises(
        OperatorHistoryError,
        match="persisted live state is invalid",
    ):
        history.get_live_session("live-001")


def test_adds_live_schema_to_existing_replay_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE replay_sessions (
                run_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                time_scale REAL,
                event_count INTEGER NOT NULL,
                failure_code TEXT,
                failure_message TEXT
            );

            CREATE TABLE prediction_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence),
                FOREIGN KEY (run_id)
                    REFERENCES replay_sessions(run_id)
                    ON DELETE CASCADE
            );
            """
        )

        connection.execute(
            """
            INSERT INTO replay_sessions (
                run_id,
                source_id,
                source_sha256,
                state,
                time_scale,
                event_count,
                failure_code,
                failure_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-replay",
                "capture.pcap",
                "a" * 64,
                "completed",
                1.0,
                0,
                None,
                None,
            ),
        )

    history = SqliteOperatorHistory(path)

    replay = history.get_replay("legacy-replay")

    assert replay is not None
    assert replay.state is ReplayState.COMPLETED

    history.save_live_session(_live_record())

    assert history.get_live_session("live-001") == _live_record()


def test_creates_history_database_with_owner_only_permissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"

    SqliteOperatorHistory(path)

    assert path.stat().st_mode & 0o777 == 0o600


def test_tightens_existing_history_database_permissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE legacy_state (
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO legacy_state (value)
            VALUES ('preserved')
            """
        )

    path.chmod(0o644)

    SqliteOperatorHistory(path)

    assert path.stat().st_mode & 0o777 == 0o600

    with sqlite3.connect(path) as connection:
        value = connection.execute(
            """
            SELECT value
            FROM legacy_state
            """
        ).fetchone()

    assert value == ("preserved",)


def test_creates_new_history_parent_as_owner_only(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private-history"
    path = parent / "operator.sqlite3"

    SqliteOperatorHistory(path)

    assert parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


def test_does_not_change_existing_history_parent_permissions(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "shared-state"
    parent.mkdir()
    parent.chmod(0o755)

    path = parent / "operator.sqlite3"

    SqliteOperatorHistory(path)

    assert parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o600
