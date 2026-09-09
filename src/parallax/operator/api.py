"""FastAPI boundary for the Parallax operator service."""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from parallax.operator.service import (
    OperatorReplayNotFoundError,
    OperatorReplayService,
    OperatorServiceError,
)


class StartReplayRequest(BaseModel):
    """Request to start one PCAP replay."""

    capture: str
    time_scale: float | None = 1.0


def create_operator_app(service: OperatorReplayService) -> FastAPI:
    """Create the operator HTTP application around an initialized service."""
    app = FastAPI(
        title="Parallax Operator API",
        version="1",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

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

    return app
