"""FastAPI boundary for the Parallax operator service."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from parallax.operator.service import (
    OperatorEventBatch,
    OperatorEventCursorError,
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


class StartReplayRequest(BaseModel):
    """Request to start one PCAP replay."""

    capture: str
    time_scale: float | None = 1.0


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


def create_operator_app(service: OperatorReplayService) -> FastAPI:
    """Create the operator HTTP application around an initialized service."""
    app = FastAPI(
        title="Parallax Operator API",
        version="1",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

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
