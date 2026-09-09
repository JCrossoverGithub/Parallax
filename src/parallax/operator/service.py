"""In-memory operator service for controlled Parallax replay sessions."""

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock, Thread
from uuid import uuid4

from parallax.modeling.runtime import PrototypeRuntime
from parallax.replay import (
    ReplayConfiguration,
    ReplayControl,
    ReplaySession,
    ReplaySessionId,
    ReplayState,
    iter_pcap_replay_entries,
)
from parallax.runtime import (
    PacketPredictionPipeline,
    RuntimePredictionEvent,
    RuntimeScorer,
    run_packet_prediction_replay,
)


class OperatorServiceError(ValueError):
    """Raised when an operator request cannot be satisfied."""


class OperatorReplayNotFoundError(OperatorServiceError):
    """Raised when an operator replay ID does not exist."""


@dataclass(frozen=True, slots=True)
class OperatorModelIdentity:
    """Artifact identities exposed to an operator without filesystem paths."""

    model_bundle_sha256: str
    calibration_artifact_sha256: str
    feature_artifact_sha256: str
    split_manifest_sha256: str

    @classmethod
    def from_runtime(cls, runtime: PrototypeRuntime) -> "OperatorModelIdentity":
        """Build the public artifact identity from a verified runtime."""
        return cls(
            model_bundle_sha256=runtime.bundle.sha256,
            calibration_artifact_sha256=runtime.calibration.sha256,
            feature_artifact_sha256=runtime.bundle.feature_artifact_sha256,
            split_manifest_sha256=runtime.bundle.split_manifest_sha256,
        )

    def as_dict(self) -> dict[str, str]:
        """Return JSON-compatible model provenance."""
        return {
            "model_bundle_sha256": self.model_bundle_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class OperatorReplaySnapshot:
    """Current operator-visible state for one replay."""

    run_id: str
    source_id: str
    source_sha256: str
    state: ReplayState
    time_scale: float | None
    event_count: int
    retained_event_count: int
    failure_code: str | None
    failure_message: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible replay snapshot."""
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
            "retained_event_count": self.retained_event_count,
            "failure": failure,
        }


@dataclass(slots=True)
class _ReplayRecord:
    session: ReplaySession
    control: ReplayControl
    events: deque[RuntimePredictionEvent]
    total_events: int = 0


class OperatorReplayService:
    """Own replay execution and bounded operator-visible in-memory state."""

    def __init__(
        self,
        *,
        capture_root: str | Path,
        scorer: RuntimeScorer,
        model_identity: OperatorModelIdentity,
        event_history_limit: int = 10_000,
    ) -> None:
        if event_history_limit < 1:
            raise OperatorServiceError("event history limit must be positive")

        self._capture_root = Path(capture_root)
        self._scorer = scorer
        self._model_identity = model_identity
        self._event_history_limit = event_history_limit
        self._records: dict[str, _ReplayRecord] = {}
        self._lock = RLock()

    def health(self) -> dict[str, object]:
        """Return service and active-model status."""
        with self._lock:
            active = sum(
                record.session.state
                in {
                    ReplayState.CREATED,
                    ReplayState.RUNNING,
                    ReplayState.PAUSED,
                }
                for record in self._records.values()
            )

            return {
                "status": "ok",
                "active_model": self._model_identity.as_dict(),
                "sessions": {
                    "total": len(self._records),
                    "active": active,
                },
            }

    def start_replay(
        self,
        capture_name: str,
        *,
        time_scale: float | None = 1.0,
    ) -> OperatorReplaySnapshot:
        """Start one controlled replay in a daemon worker thread."""
        source = self._resolve_capture(capture_name)

        try:
            configuration = ReplayConfiguration(time_scale=time_scale)
        except ValueError as error:
            raise OperatorServiceError(str(error)) from error

        run_id = str(uuid4())
        session = ReplaySession(
            session_id=ReplaySessionId.parse(run_id),
            source_id=source.name,
            source_sha256=_sha256_file(source),
            configuration=configuration,
        )
        control = ReplayControl()
        record = _ReplayRecord(
            session=session,
            control=control,
            events=deque(maxlen=self._event_history_limit),
        )

        with self._lock:
            self._records[run_id] = record
            snapshot = self._snapshot(record, run_id=run_id)

        thread = Thread(
            target=self._execute_replay,
            args=(run_id, source, configuration),
            name=f"parallax-replay-{run_id[:8]}",
            daemon=True,
        )
        thread.start()

        return snapshot

    def get_replay(self, run_id: str) -> OperatorReplaySnapshot:
        """Return current state for one replay."""
        with self._lock:
            record = self._require_record(run_id)
            return self._snapshot(record, run_id=run_id)

    def get_events(self, run_id: str) -> tuple[dict[str, object], ...]:
        """Return the retained ordered prediction-event history."""
        with self._lock:
            record = self._require_record(run_id)
            return tuple(event.as_dict() for event in record.events)

    def _execute_replay(
        self,
        run_id: str,
        source: Path,
        configuration: ReplayConfiguration,
    ) -> None:
        with self._lock:
            record = self._require_record(run_id)
            control = record.control
            session = record.session

        pipeline = PacketPredictionPipeline(
            run_id=run_id,
            capture_name=source.name,
            scorer=self._scorer,
        )

        def handle_session(current: ReplaySession) -> None:
            if current.state not in {
                ReplayState.COMPLETED,
                ReplayState.FAILED,
                ReplayState.CANCELLED,
            }:
                self._update_session(run_id, current)

        result = run_packet_prediction_replay(
            session,
            iter_pcap_replay_entries(
                source,
                configuration=configuration,
            ),
            pipeline=pipeline,
            control=control,
            handle_event=lambda event: self._append_event(run_id, event),
            handle_session=handle_session,
        )

        # Publish a terminal session only after replay integration has completed.
        # For COMPLETED, this guarantees the final eligible runtime window has
        # already been flushed into operator-visible event history.
        self._update_session(run_id, result)

    def _append_event(
        self,
        run_id: str,
        event: RuntimePredictionEvent,
    ) -> None:
        with self._lock:
            record = self._require_record(run_id)
            record.events.append(event)
            record.total_events += 1

    def _update_session(
        self,
        run_id: str,
        session: ReplaySession,
    ) -> None:
        with self._lock:
            self._require_record(run_id).session = session

    def _resolve_capture(self, capture_name: str) -> Path:
        if not capture_name or Path(capture_name).name != capture_name:
            raise OperatorServiceError("capture must be specified by filename only")

        source = self._capture_root / capture_name

        if not source.is_file():
            raise OperatorServiceError(f"capture {capture_name!r} does not exist")

        return source

    def _require_record(self, run_id: str) -> _ReplayRecord:
        try:
            return self._records[run_id]
        except KeyError as error:
            raise OperatorReplayNotFoundError(f"replay {run_id!r} does not exist") from error

    @staticmethod
    def _snapshot(
        record: _ReplayRecord,
        *,
        run_id: str,
    ) -> OperatorReplaySnapshot:
        failure = record.session.failure

        return OperatorReplaySnapshot(
            run_id=run_id,
            source_id=record.session.source_id,
            source_sha256=record.session.source_sha256,
            state=record.session.state,
            time_scale=record.session.configuration.time_scale,
            event_count=record.total_events,
            retained_event_count=len(record.events),
            failure_code=None if failure is None else failure.code,
            failure_message=None if failure is None else failure.message,
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()
