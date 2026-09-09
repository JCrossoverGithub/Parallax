"""FastAPI boundary for the Parallax operator service."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from parallax.operator.live import (
    OperatorLiveSession,
    OperatorLiveState,
)
from parallax.operator.service import (
    OperatorEventBatch,
    OperatorEventCursorError,
    OperatorLiveConflictError,
    OperatorLiveEventBatch,
    OperatorLiveNotFoundError,
    OperatorReplayConflictError,
    OperatorReplayNotFoundError,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.replay import ReplayState

_TERMINAL_STATES = frozenset(
    {
        ReplayState.COMPLETED,
        ReplayState.FAILED,
        ReplayState.CANCELLED,
    }
)


_LIVE_TERMINAL_STATES = frozenset(
    {
        OperatorLiveState.COMPLETED,
        OperatorLiveState.FAILED,
    }
)


class StartReplayRequest(BaseModel):
    """Request to start one PCAP replay."""

    capture: str
    time_scale: float | None = 1.0


class StartLiveRequest(BaseModel):
    """Request to start one live capture session."""

    interface: str


def _live_session_payload(
    session: OperatorLiveSession,
) -> dict[str, object]:
    """Serialize one live operator session for the HTTP boundary."""
    failure = None if session.failure is None else session.failure.as_dict()

    return {
        "run_id": session.run_id,
        "state": session.state.value,
        "configuration": {
            "interface": session.configuration.interface,
            "stale_after_seconds": (session.configuration.stale_after_seconds),
            "max_tracked_flows": (session.configuration.max_tracked_flows),
        },
        "failure": failure,
    }


def _control_request(
    run_id: str,
    action: str,
    operation: Callable[[str], None],
) -> dict[str, object]:
    try:
        operation(run_id)
    except OperatorReplayNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except OperatorReplayConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    return {
        "run_id": run_id,
        "action": action,
        "accepted": True,
    }


def _parse_stream_cursor(request: Request, after: int) -> int:
    last_event_id = request.headers.get("last-event-id")

    if last_event_id is None:
        return after

    try:
        cursor = int(last_event_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be an integer",
        ) from error

    if cursor < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must not be negative",
        )

    return cursor


def _prediction_sse(
    sequence: int,
    payload: dict[str, object],
) -> str:
    data = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    )

    return f"id: {sequence}\nevent: prediction\ndata: {data}\n\n"


def _terminal_sse(
    run_id: str,
    state: ReplayState,
) -> str:
    data = json.dumps(
        {
            "run_id": run_id,
            "state": state.value,
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    return f"event: replay-terminal\ndata: {data}\n\n"


def _live_terminal_sse(
    run_id: str,
    state: OperatorLiveState,
) -> str:
    data = json.dumps(
        {
            "run_id": run_id,
            "state": state.value,
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    return f"event: live-terminal\ndata: {data}\n\n"


def _stream_error_sse(message: str) -> str:
    data = json.dumps(
        {"detail": message},
        separators=(",", ":"),
        sort_keys=True,
    )

    return f"event: stream-error\ndata: {data}\n\n"


async def _stream_events(
    service: OperatorReplayService,
    run_id: str,
    initial_batch: OperatorEventBatch,
) -> AsyncIterator[str]:
    batch = initial_batch

    while True:
        for event in batch.events:
            yield _prediction_sse(
                event.sequence,
                event.payload,
            )

        cursor = batch.last_sequence

        if batch.state in _TERMINAL_STATES:
            yield _terminal_sse(run_id, batch.state)
            return

        await asyncio.sleep(0.05)

        try:
            batch = service.get_event_batch(
                run_id,
                after_sequence=cursor,
            )
        except OperatorEventCursorError as error:
            yield _stream_error_sse(str(error))
            return


async def _stream_live_events(
    service: OperatorReplayService,
    run_id: str,
    initial_batch: OperatorLiveEventBatch,
) -> AsyncIterator[str]:
    batch = initial_batch

    while True:
        for event in batch.events:
            yield _prediction_sse(
                event.sequence,
                event.payload,
            )

        cursor = batch.last_sequence

        if batch.state in _LIVE_TERMINAL_STATES:
            yield _live_terminal_sse(
                run_id,
                batch.state,
            )
            return

        await asyncio.sleep(0.05)

        try:
            batch = service.get_live_event_batch(
                run_id,
                after_sequence=cursor,
            )
        except OperatorEventCursorError as error:
            yield _stream_error_sse(str(error))
            return


def create_operator_app(service: OperatorReplayService) -> FastAPI:
    """Create the operator HTTP application around an initialized service."""
    app = FastAPI(
        title="Parallax Operator API",
        version="1",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.get("/api/v1/live/interfaces")
    def list_live_interfaces() -> dict[str, object]:
        try:
            interfaces = service.list_live_interfaces()
        except OperatorServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

        return {
            "interfaces": [
                {
                    "index": interface.index,
                    "name": interface.name,
                }
                for interface in interfaces
            ]
        }

    @app.post(
        "/api/v1/live",
        status_code=status.HTTP_201_CREATED,
    )
    def start_live(
        request: StartLiveRequest,
    ) -> dict[str, object]:
        try:
            session = service.start_live(request.interface)
        except OperatorLiveConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        except OperatorServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

        return _live_session_payload(session)

    @app.get("/api/v1/live/active")
    def get_active_live() -> dict[str, object] | None:
        session = service.get_active_live()

        if session is None:
            return None

        return _live_session_payload(session)

    @app.get("/api/v1/live/{run_id}")
    def get_live(run_id: str) -> dict[str, object]:
        try:
            session = service.get_live(run_id)
        except OperatorLiveNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return _live_session_payload(session)

    @app.get("/api/v1/live/{run_id}/events")
    def get_live_events(
        run_id: str,
    ) -> dict[str, object]:
        try:
            snapshot = service.get_live_event_snapshot(
                run_id,
            )
        except OperatorLiveNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return {
            "run_id": snapshot.run_id,
            "events": list(snapshot.events),
            "last_sequence": snapshot.last_sequence,
            "state": snapshot.state.value,
        }

    @app.post(
        "/api/v1/live/{run_id}/stop",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def stop_live(run_id: str) -> dict[str, object]:
        try:
            session = service.stop_live(run_id)
        except OperatorLiveNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except OperatorLiveConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error

        return {
            "run_id": run_id,
            "action": "stop",
            "accepted": True,
            "state": session.state.value,
        }

    @app.get("/api/v1/live/{run_id}/stream")
    def stream_live(
        run_id: str,
        request: Request,
        after: int = 0,
    ) -> StreamingResponse:
        cursor = _parse_stream_cursor(request, after)

        try:
            initial_batch = service.get_live_event_batch(
                run_id,
                after_sequence=cursor,
            )
        except OperatorLiveNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except OperatorEventCursorError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error

        return StreamingResponse(
            _stream_live_events(
                service,
                run_id,
                initial_batch,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/history")
    def list_history(
        limit: int = 100,
    ) -> dict[str, object]:
        try:
            records = service.list_history(limit=limit)
        except OperatorServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

        return {"replays": [record.as_dict() for record in records]}

    @app.get("/api/v1/history/{run_id}")
    def get_history_replay(
        run_id: str,
    ) -> dict[str, object]:
        try:
            return service.get_history_replay(run_id).as_dict()
        except OperatorReplayNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @app.get("/api/v1/history/{run_id}/events")
    def get_history_events(
        run_id: str,
    ) -> dict[str, object]:
        try:
            events = service.get_history_events(run_id)
        except OperatorReplayNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return {
            "run_id": run_id,
            "events": list(events),
        }

    @app.post(
        "/api/v1/replays",
        status_code=status.HTTP_201_CREATED,
    )
    def start_replay(request: StartReplayRequest) -> dict[str, object]:
        try:
            return service.start_replay(
                request.capture,
                time_scale=request.time_scale,
            ).as_dict()
        except OperatorServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    @app.get("/api/v1/replays/{run_id}")
    def get_replay(run_id: str) -> dict[str, object]:
        try:
            return service.get_replay(run_id).as_dict()
        except OperatorReplayNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @app.get("/api/v1/replays/{run_id}/events")
    def get_replay_events(run_id: str) -> dict[str, object]:
        try:
            events = service.get_events(run_id)
        except OperatorReplayNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return {
            "run_id": run_id,
            "events": list(events),
        }

    @app.post(
        "/api/v1/replays/{run_id}/pause",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def pause_replay(run_id: str) -> dict[str, object]:
        return _control_request(
            run_id,
            "pause",
            service.pause_replay,
        )

    @app.post(
        "/api/v1/replays/{run_id}/resume",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_replay(run_id: str) -> dict[str, object]:
        return _control_request(
            run_id,
            "resume",
            service.resume_replay,
        )

    @app.post(
        "/api/v1/replays/{run_id}/cancel",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_replay(run_id: str) -> dict[str, object]:
        return _control_request(
            run_id,
            "cancel",
            service.cancel_replay,
        )

    @app.get("/api/v1/replays/{run_id}/stream")
    def stream_replay(
        run_id: str,
        request: Request,
        after: int = 0,
    ) -> StreamingResponse:
        cursor = _parse_stream_cursor(request, after)

        try:
            initial_batch = service.get_event_batch(
                run_id,
                after_sequence=cursor,
            )
        except OperatorReplayNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except OperatorEventCursorError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error

        return StreamingResponse(
            _stream_events(
                service,
                run_id,
                initial_batch,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
