"""In-memory operator service for controlled Parallax replay sessions."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

from parallax.modeling.runtime import PrototypeRuntime
from parallax.operator.history import (
    OperatorHistoryError,
    OperatorHistoryRecord,
    OperatorLiveHistoryRecord,
    SqliteOperatorHistory,
)
from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveSession,
    OperatorLiveSessionError,
    OperatorLiveState,
)
from parallax.replay import (
    ReplayConfiguration,
    ReplayControl,
    ReplayControlError,
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
from parallax.sensor import (
    CaptureInterface,
    SensorInterfaceError,
    list_capture_interfaces,
    resolve_capture_interface,
)

_TERMINAL_STATES = frozenset(
    {
        ReplayState.COMPLETED,
        ReplayState.FAILED,
        ReplayState.CANCELLED,
    }
)


class OperatorServiceError(ValueError):
    """Raised when an operator request cannot be satisfied."""


class OperatorReplayNotFoundError(OperatorServiceError):
    """Raised when an operator replay ID does not exist."""


class OperatorReplayConflictError(OperatorServiceError):
    """Raised when a replay cannot accept an operator control request."""


class OperatorEventCursorError(OperatorServiceError):
    """Raised when an event-stream cursor cannot be satisfied."""


class OperatorLiveNotFoundError(OperatorServiceError):
    """Raised when an operator live-session ID does not exist."""


class OperatorLiveConflictError(OperatorServiceError):
    """Raised when a live sensor request conflicts with current state."""


OperatorLiveExecutor = Callable[
    [
        str,
        OperatorLiveConfiguration,
        Event,
        Callable[[RuntimePredictionEvent], None],
    ],
    object,
]


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


@dataclass(frozen=True, slots=True)
class OperatorEventRecord:
    """One sequenced operator-visible prediction event."""

    sequence: int
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class OperatorEventBatch:
    """Prediction events available after one sequence cursor."""

    run_id: str
    events: tuple[OperatorEventRecord, ...]
    last_sequence: int
    state: ReplayState


@dataclass(frozen=True, slots=True)
class OperatorLiveEventBatch:
    """Prediction events available after one live-session cursor."""

    run_id: str
    events: tuple[OperatorEventRecord, ...]
    last_sequence: int
    state: OperatorLiveState


@dataclass(frozen=True, slots=True)
class OperatorLiveEventSnapshot:
    """Current retained live events and their global sequence cursor."""

    run_id: str
    events: tuple[dict[str, object], ...]
    last_sequence: int
    state: OperatorLiveState


@dataclass(slots=True)
class _ReplayRecord:
    session: ReplaySession
    control: ReplayControl
    events: deque[RuntimePredictionEvent]
    total_events: int = 0


@dataclass(slots=True)
class _LiveRecord:
    session: OperatorLiveSession
    stop_event: Event
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
        history: SqliteOperatorHistory | None = None,
        event_history_limit: int = 10_000,
        live_executor: OperatorLiveExecutor | None = None,
        live_interface_lister: Callable[[], tuple[CaptureInterface, ...]] = list_capture_interfaces,
        live_interface_resolver: Callable[[str], CaptureInterface] = resolve_capture_interface,
        live_stale_after_seconds: float = 120.0,
        live_max_tracked_flows: int = 4_096,
    ) -> None:
        if event_history_limit < 1:
            raise OperatorServiceError("event history limit must be positive")

        self._capture_root = Path(capture_root)
        self._scorer = scorer
        self._model_identity = model_identity
        self._history = history
        self._event_history_limit = event_history_limit
        self._live_executor = live_executor
        self._live_interface_lister = live_interface_lister
        self._live_interface_resolver = live_interface_resolver
        self._live_stale_after_seconds = live_stale_after_seconds
        self._live_max_tracked_flows = live_max_tracked_flows
        self._records: dict[str, _ReplayRecord] = {}
        self._live_records: dict[str, _LiveRecord] = {}
        self._lock = RLock()

        if self._history is not None:
            self._history.fail_interrupted_live_sessions()

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

    def list_live_interfaces(
        self,
    ) -> tuple[CaptureInterface, ...]:
        """Return capture interfaces available to the live operator."""
        try:
            return self._live_interface_lister()
        except SensorInterfaceError as error:
            raise OperatorServiceError(str(error)) from error

    def start_live(
        self,
        interface: str,
    ) -> OperatorLiveSession:
        """Create and start one owned live sensor session."""
        if self._live_executor is None:
            raise OperatorServiceError("live capture execution is not configured")

        try:
            resolved_interface = self._live_interface_resolver(interface)
            configuration = OperatorLiveConfiguration(
                interface=resolved_interface.name,
                stale_after_seconds=self._live_stale_after_seconds,
                max_tracked_flows=self._live_max_tracked_flows,
            )
        except (
            SensorInterfaceError,
            OperatorLiveSessionError,
        ) as error:
            raise OperatorServiceError(str(error)) from error

        run_id = str(uuid4())
        session = OperatorLiveSession(
            run_id=run_id,
            configuration=configuration,
        )
        record = _LiveRecord(
            session=session,
            stop_event=Event(),
            events=deque(maxlen=self._event_history_limit),
        )

        with self._lock:
            if any(
                not current.session.state.is_terminal for current in self._live_records.values()
            ):
                raise OperatorLiveConflictError("a live sensor session is already active")

            self._live_records[run_id] = record
            self._persist_live(
                record,
                run_id=run_id,
            )

        thread = Thread(
            target=self._execute_live,
            args=(run_id,),
            name=f"parallax-live-{run_id[:8]}",
            daemon=True,
        )
        thread.start()

        return session

    def get_active_live(
        self,
    ) -> OperatorLiveSession | None:
        """Return the currently active live session, if one exists."""
        with self._lock:
            for record in self._live_records.values():
                if not record.session.state.is_terminal:
                    return record.session

        return None

    def get_live(
        self,
        run_id: str,
    ) -> OperatorLiveSession:
        """Return current lifecycle state for one live session."""
        with self._lock:
            return self._require_live_record(run_id).session

    def get_live_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        """Return retained prediction events for one live session."""
        return self.get_live_event_snapshot(run_id).events

    def get_live_event_snapshot(
        self,
        run_id: str,
    ) -> OperatorLiveEventSnapshot:
        """Return retained events with their current global sequence cursor."""
        with self._lock:
            record = self._require_live_record(run_id)

            return OperatorLiveEventSnapshot(
                run_id=run_id,
                events=tuple(event.as_dict() for event in record.events),
                last_sequence=record.total_events,
                state=record.session.state,
            )

    def get_live_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorLiveEventBatch:
        """Return retained live events following one stream cursor."""
        if after_sequence < 0:
            raise OperatorEventCursorError("event sequence cursor must not be negative")

        with self._lock:
            record = self._require_live_record(run_id)

            if after_sequence > record.total_events:
                raise OperatorEventCursorError("event sequence cursor is ahead of the live session")

            first_sequence = record.total_events - len(record.events) + 1

            if record.events and after_sequence < first_sequence - 1:
                raise OperatorEventCursorError("requested events are no longer retained")

            events = tuple(
                OperatorEventRecord(
                    sequence=sequence,
                    payload=event.as_dict(),
                )
                for sequence, event in enumerate(
                    record.events,
                    start=first_sequence,
                )
                if sequence > after_sequence
            )

            return OperatorLiveEventBatch(
                run_id=run_id,
                events=events,
                last_sequence=record.total_events,
                state=record.session.state,
            )

    def stop_live(
        self,
        run_id: str,
    ) -> OperatorLiveSession:
        """Request orderly termination of one live sensor session."""
        with self._lock:
            record = self._require_live_record(run_id)

            if record.session.state.is_terminal:
                raise OperatorLiveConflictError(f"live session {run_id!r} is already terminal")

            if record.session.state in {
                OperatorLiveState.STARTING,
                OperatorLiveState.RUNNING,
            }:
                record.session = record.session.transition(OperatorLiveState.STOPPING)

            record.stop_event.set()
            self._persist_live(
                record,
                run_id=run_id,
            )
            return record.session

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

        self._persist_replay(record, run_id=run_id)

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

    def list_history(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OperatorHistoryRecord, ...]:
        """Return persisted replay history newest first."""
        if self._history is None:
            return ()

        try:
            return self._history.list_replays(limit=limit)
        except OperatorHistoryError as error:
            raise OperatorServiceError(str(error)) from error

    def get_history_replay(
        self,
        run_id: str,
    ) -> OperatorHistoryRecord:
        """Return one persisted replay summary."""
        if self._history is None:
            raise OperatorReplayNotFoundError(f"persisted replay {run_id!r} does not exist")

        record = self._history.get_replay(run_id)

        if record is None:
            raise OperatorReplayNotFoundError(f"persisted replay {run_id!r} does not exist")

        return record

    def get_history_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        """Return persisted prediction events for one replay."""
        self.get_history_replay(run_id)

        assert self._history is not None
        return self._history.get_events(run_id)

    def list_live_history(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OperatorLiveHistoryRecord, ...]:
        """Return persisted live-session history newest first."""
        if self._history is None:
            return ()

        try:
            return self._history.list_live_sessions(limit=limit)
        except OperatorHistoryError as error:
            raise OperatorServiceError(str(error)) from error

    def get_history_live(
        self,
        run_id: str,
    ) -> OperatorLiveHistoryRecord:
        """Return one persisted live-session summary."""
        if self._history is None:
            raise OperatorLiveNotFoundError(f"persisted live session {run_id!r} does not exist")

        record = self._history.get_live_session(run_id)

        if record is None:
            raise OperatorLiveNotFoundError(f"persisted live session {run_id!r} does not exist")

        return record

    def get_history_live_events(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        """Return persisted prediction events for one live session."""
        self.get_history_live(run_id)

        assert self._history is not None

        return self._history.get_live_events(run_id)

    def get_event_batch(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> OperatorEventBatch:
        """Return retained events following an inclusive stream cursor."""
        if after_sequence < 0:
            raise OperatorEventCursorError("event sequence cursor must not be negative")

        with self._lock:
            record = self._require_record(run_id)

            if after_sequence > record.total_events:
                raise OperatorEventCursorError("event sequence cursor is ahead of the replay")

            first_sequence = record.total_events - len(record.events) + 1

            if record.events and after_sequence < first_sequence - 1:
                raise OperatorEventCursorError("requested events are no longer retained")

            events = tuple(
                OperatorEventRecord(
                    sequence=sequence,
                    payload=event.as_dict(),
                )
                for sequence, event in enumerate(
                    record.events,
                    start=first_sequence,
                )
                if sequence > after_sequence
            )

            return OperatorEventBatch(
                run_id=run_id,
                events=events,
                last_sequence=record.total_events,
                state=record.session.state,
            )

    def pause_replay(self, run_id: str) -> None:
        """Request that a running replay pause."""
        self._request_control(run_id, ReplayControl.pause)

    def resume_replay(self, run_id: str) -> None:
        """Request that a paused replay resume."""
        self._request_control(run_id, ReplayControl.resume)

    def cancel_replay(self, run_id: str) -> None:
        """Request cancellation of a nonterminal replay."""
        self._request_control(run_id, ReplayControl.cancel)

    def _request_control(
        self,
        run_id: str,
        request: Callable[[ReplayControl], None],
    ) -> None:
        with self._lock:
            record = self._require_record(run_id)

            if record.session.state in _TERMINAL_STATES:
                raise OperatorReplayConflictError(f"replay {run_id!r} is already terminal")

            try:
                request(record.control)
            except ReplayControlError as error:
                raise OperatorReplayConflictError(str(error)) from error

    def _append_live_event(
        self,
        run_id: str,
        event: RuntimePredictionEvent,
    ) -> None:
        with self._lock:
            record = self._require_live_record(run_id)
            record.events.append(event)
            record.total_events += 1
            sequence = record.total_events

            if self._history is not None:
                self._history.save_live_event(
                    run_id,
                    sequence,
                    event.as_dict(),
                )

    def _execute_live(
        self,
        run_id: str,
    ) -> None:
        with self._lock:
            record = self._require_live_record(run_id)

            if record.session.state is OperatorLiveState.STOPPING:
                record.session = record.session.transition(OperatorLiveState.COMPLETED)
                self._persist_live(
                    record,
                    run_id=run_id,
                )
                return

            record.session = record.session.transition(OperatorLiveState.RUNNING)
            self._persist_live(
                record,
                run_id=run_id,
            )
            configuration = record.session.configuration
            stop_event = record.stop_event

        executor = self._live_executor
        assert executor is not None

        try:
            executor(
                run_id,
                configuration,
                stop_event,
                lambda event: self._append_live_event(
                    run_id,
                    event,
                ),
            )
        except Exception as error:
            with self._lock:
                record = self._require_live_record(run_id)
                record.session = record.session.fail(
                    code="live_execution_error",
                    message=str(error),
                )
                self._persist_live(
                    record,
                    run_id=run_id,
                )
            return

        with self._lock:
            record = self._require_live_record(run_id)

            if record.session.state is OperatorLiveState.RUNNING:
                record.session = record.session.transition(OperatorLiveState.STOPPING)

            record.session = record.session.transition(OperatorLiveState.COMPLETED)
            self._persist_live(
                record,
                run_id=run_id,
            )

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
            capture_id=source.name,
            scorer=self._scorer,
        )

        def handle_session(current: ReplaySession) -> None:
            if current.state not in _TERMINAL_STATES:
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
            sequence = record.total_events
            history_record = self._make_history_record(
                record,
                run_id=run_id,
            )

        if self._history is not None:
            self._history.save_event(
                run_id,
                sequence,
                event.as_dict(),
            )
            self._history.save_replay(history_record)

    def _update_session(
        self,
        run_id: str,
        session: ReplaySession,
    ) -> None:
        with self._lock:
            record = self._require_record(run_id)
            record.session = session
            history_record = self._make_history_record(
                record,
                run_id=run_id,
            )

        if self._history is not None:
            self._history.save_replay(history_record)

    def _persist_live(
        self,
        record: _LiveRecord,
        *,
        run_id: str,
    ) -> None:
        if self._history is not None:
            self._history.save_live_session(
                self._make_live_history_record(
                    record,
                    run_id=run_id,
                )
            )

    @staticmethod
    def _make_live_history_record(
        record: _LiveRecord,
        *,
        run_id: str,
    ) -> OperatorLiveHistoryRecord:
        failure = record.session.failure
        configuration = record.session.configuration

        return OperatorLiveHistoryRecord(
            run_id=run_id,
            interface=configuration.interface,
            state=record.session.state,
            stale_after_seconds=(configuration.stale_after_seconds),
            max_tracked_flows=(configuration.max_tracked_flows),
            event_count=record.total_events,
            failure_code=(None if failure is None else failure.code),
            failure_message=(None if failure is None else failure.message),
        )

    def _persist_replay(
        self,
        record: _ReplayRecord,
        *,
        run_id: str,
    ) -> None:
        if self._history is not None:
            self._history.save_replay(
                self._make_history_record(
                    record,
                    run_id=run_id,
                )
            )

    @staticmethod
    def _make_history_record(
        record: _ReplayRecord,
        *,
        run_id: str,
    ) -> OperatorHistoryRecord:
        failure = record.session.failure

        return OperatorHistoryRecord(
            run_id=run_id,
            source_id=record.session.source_id,
            source_sha256=record.session.source_sha256,
            state=record.session.state,
            time_scale=record.session.configuration.time_scale,
            event_count=record.total_events,
            failure_code=None if failure is None else failure.code,
            failure_message=None if failure is None else failure.message,
        )

    def _resolve_capture(self, capture_name: str) -> Path:
        if not capture_name or Path(capture_name).name != capture_name:
            raise OperatorServiceError("capture must be specified by filename only")

        source = self._capture_root / capture_name

        if not source.is_file():
            raise OperatorServiceError(f"capture {capture_name!r} does not exist")

        return source

    def _require_live_record(
        self,
        run_id: str,
    ) -> _LiveRecord:
        try:
            return self._live_records[run_id]
        except KeyError as error:
            raise OperatorLiveNotFoundError(f"live session {run_id!r} does not exist") from error

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
