"""Durable local history for Parallax operator replay sessions."""

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from parallax.operator.live import OperatorLiveState
from parallax.replay import ReplayState


class OperatorHistoryError(ValueError):
    """Raised when persisted operator history is invalid."""


@dataclass(frozen=True, slots=True)
class OperatorHistoryRecord:
    """Persisted summary of one replay session."""

    run_id: str
    source_id: str
    source_sha256: str
    state: ReplayState
    time_scale: float | None
    event_count: int
    failure_code: str | None
    failure_message: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible replay-history record."""
        failure: dict[str, str] | None = None

        if self.failure_code is not None and self.failure_message is not None:
            failure = {
                "code": self.failure_code,
                "message": self.failure_message,
            }

        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "state": self.state.value,
            "time_scale": self.time_scale,
            "event_count": self.event_count,
            "failure": failure,
        }


@dataclass(frozen=True, slots=True)
class OperatorLiveHistoryRecord:
    """Persisted summary of one live sensor session."""

    run_id: str
    interface: str
    state: OperatorLiveState
    stale_after_seconds: float
    max_tracked_flows: int
    event_count: int
    failure_code: str | None
    failure_message: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible live-history record."""
        failure: dict[str, str] | None = None

        if self.failure_code is not None and self.failure_message is not None:
            failure = {
                "code": self.failure_code,
                "message": self.failure_message,
            }

        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "configuration": {
                "interface": self.interface,
                "stale_after_seconds": self.stale_after_seconds,
                "max_tracked_flows": self.max_tracked_flows,
            },
            "event_count": self.event_count,
            "failure": failure,
        }


class SqliteOperatorHistory:
    """Persist replay and live operator history in local SQLite."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

        if self._path.exists() and self._path.is_dir():
            raise OperatorHistoryError("operator history path must be a database file")

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )
        self._prepare_database_file()
        self._initialize()

    def _prepare_database_file(self) -> None:
        """Create or tighten the history database as owner-only state."""
        descriptor = os.open(
            self._path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )

        try:
            os.fchmod(
                descriptor,
                0o600,
            )
        finally:
            os.close(descriptor)

    def save_replay(
        self,
        record: OperatorHistoryRecord,
    ) -> None:
        """Insert or update one replay summary."""
        with self._connect() as connection:
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
                ON CONFLICT(run_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    source_sha256 = excluded.source_sha256,
                    state = excluded.state,
                    time_scale = excluded.time_scale,
                    event_count = excluded.event_count,
                    failure_code = excluded.failure_code,
                    failure_message = excluded.failure_message
                """,
                (
                    record.run_id,
                    record.source_id,
                    record.source_sha256,
                    record.state.value,
                    record.time_scale,
                    record.event_count,
                    record.failure_code,
                    record.failure_message,
                ),
            )

    def save_event(
        self,
        run_id: str,
        sequence: int,
        payload: dict[str, object],
    ) -> None:
        """Persist one ordered runtime prediction event."""
        if sequence < 1:
            raise OperatorHistoryError("persisted event sequence must be positive")

        serialized = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )

        with self._connect() as connection:
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
                    run_id,
                    sequence,
                    serialized,
                ),
            )

    def get_replay(
        self,
        run_id: str,
    ) -> OperatorHistoryRecord | None:
        """Load one persisted replay summary."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    run_id,
                    source_id,
                    source_sha256,
                    state,
                    time_scale,
                    event_count,
                    failure_code,
                    failure_message
                FROM replay_sessions
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return _history_record(row)

    def list_replays(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OperatorHistoryRecord, ...]:
        """List most recently created replay summaries."""
        if limit < 1:
            raise OperatorHistoryError("history list limit must be positive")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    run_id,
                    source_id,
                    source_sha256,
                    state,
                    time_scale,
                    event_count,
                    failure_code,
                    failure_message
                FROM replay_sessions
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return tuple(_history_record(row) for row in rows)

    def get_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        """Load all persisted prediction events for one replay."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM prediction_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()

        return tuple(_load_event_payload(str(row["payload_json"])) for row in rows)

    def save_live_session(
        self,
        record: OperatorLiveHistoryRecord,
    ) -> None:
        """Insert or update one live-session summary."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_sessions (
                    run_id,
                    interface,
                    state,
                    stale_after_seconds,
                    max_tracked_flows,
                    event_count,
                    failure_code,
                    failure_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    interface = excluded.interface,
                    state = excluded.state,
                    stale_after_seconds = excluded.stale_after_seconds,
                    max_tracked_flows = excluded.max_tracked_flows,
                    event_count = excluded.event_count,
                    failure_code = excluded.failure_code,
                    failure_message = excluded.failure_message
                """,
                (
                    record.run_id,
                    record.interface,
                    record.state.value,
                    record.stale_after_seconds,
                    record.max_tracked_flows,
                    record.event_count,
                    record.failure_code,
                    record.failure_message,
                ),
            )

    def save_live_event(
        self,
        run_id: str,
        sequence: int,
        payload: dict[str, object],
    ) -> None:
        """Persist one ordered live prediction event."""
        if sequence < 1:
            raise OperatorHistoryError("persisted live event sequence must be positive")

        serialized = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_prediction_events (
                    run_id,
                    sequence,
                    payload_json
                )
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    serialized,
                ),
            )
            connection.execute(
                """
                UPDATE live_sessions
                SET event_count = CASE
                    WHEN event_count < ? THEN ?
                    ELSE event_count
                END
                WHERE run_id = ?
                """,
                (
                    sequence,
                    sequence,
                    run_id,
                ),
            )

    def get_live_session(
        self,
        run_id: str,
    ) -> OperatorLiveHistoryRecord | None:
        """Load one persisted live-session summary."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    run_id,
                    interface,
                    state,
                    stale_after_seconds,
                    max_tracked_flows,
                    event_count,
                    failure_code,
                    failure_message
                FROM live_sessions
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return _live_history_record(row)

    def list_live_sessions(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OperatorLiveHistoryRecord, ...]:
        """List most recently created persisted live sessions."""
        if limit < 1:
            raise OperatorHistoryError("live history list limit must be positive")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    run_id,
                    interface,
                    state,
                    stale_after_seconds,
                    max_tracked_flows,
                    event_count,
                    failure_code,
                    failure_message
                FROM live_sessions
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return tuple(_live_history_record(row) for row in rows)

    def get_live_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        """Load all persisted prediction events for one live session."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM live_prediction_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()

        return tuple(_load_event_payload(str(row["payload_json"])) for row in rows)

    def fail_interrupted_live_sessions(self) -> int:
        """Mark live sessions interrupted by an operator restart as failed."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE live_sessions
                SET
                    state = ?,
                    failure_code = ?,
                    failure_message = ?
                WHERE state IN (?, ?, ?)
                """,
                (
                    OperatorLiveState.FAILED.value,
                    "operator_restart",
                    ("operator process restarted before the live session reached a terminal state"),
                    OperatorLiveState.STARTING.value,
                    OperatorLiveState.RUNNING.value,
                    OperatorLiveState.STOPPING.value,
                ),
            )

        return cursor.rowcount

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS replay_sessions (
                    run_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    time_scale REAL,
                    event_count INTEGER NOT NULL,
                    failure_code TEXT,
                    failure_message TEXT
                );

                CREATE TABLE IF NOT EXISTS prediction_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id)
                        REFERENCES replay_sessions(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    prediction_events_run_id_sequence
                ON prediction_events(run_id, sequence);

                CREATE TABLE IF NOT EXISTS live_sessions (
                    run_id TEXT PRIMARY KEY,
                    interface TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stale_after_seconds REAL NOT NULL,
                    max_tracked_flows INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    failure_code TEXT,
                    failure_message TEXT
                );

                CREATE TABLE IF NOT EXISTS live_prediction_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id)
                        REFERENCES live_sessions(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    live_prediction_events_run_id_sequence
                ON live_prediction_events(run_id, sequence);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _history_record(
    row: sqlite3.Row,
) -> OperatorHistoryRecord:
    try:
        state = ReplayState(str(row["state"]))
    except ValueError as error:
        raise OperatorHistoryError(
            f"persisted replay state is invalid: {row['state']!r}"
        ) from error

    return OperatorHistoryRecord(
        run_id=str(row["run_id"]),
        source_id=str(row["source_id"]),
        source_sha256=str(row["source_sha256"]),
        state=state,
        time_scale=(None if row["time_scale"] is None else float(row["time_scale"])),
        event_count=int(row["event_count"]),
        failure_code=(None if row["failure_code"] is None else str(row["failure_code"])),
        failure_message=(None if row["failure_message"] is None else str(row["failure_message"])),
    )


def _live_history_record(
    row: sqlite3.Row,
) -> OperatorLiveHistoryRecord:
    try:
        state = OperatorLiveState(str(row["state"]))
    except ValueError as error:
        raise OperatorHistoryError(f"persisted live state is invalid: {row['state']!r}") from error

    return OperatorLiveHistoryRecord(
        run_id=str(row["run_id"]),
        interface=str(row["interface"]),
        state=state,
        stale_after_seconds=float(row["stale_after_seconds"]),
        max_tracked_flows=int(row["max_tracked_flows"]),
        event_count=int(row["event_count"]),
        failure_code=(None if row["failure_code"] is None else str(row["failure_code"])),
        failure_message=(None if row["failure_message"] is None else str(row["failure_message"])),
    )


def _load_event_payload(
    serialized: str,
) -> dict[str, object]:
    payload = json.loads(serialized)

    if not isinstance(payload, dict):
        raise OperatorHistoryError("persisted prediction event must be a JSON object")

    return payload
