"""Controlled replay integration for the packet prediction runtime."""

from collections.abc import Callable, Iterable

from parallax.replay import (
    ReplayClock,
    ReplayControl,
    ReplayPacketEntry,
    ReplaySession,
    ReplayState,
    run_controlled_replay_schedule,
)
from parallax.runtime.events import RuntimePredictionEvent
from parallax.runtime.pipeline import PacketPredictionPipeline

RuntimeEventHandler = Callable[[RuntimePredictionEvent], None]
RuntimeSessionHandler = Callable[[ReplaySession], None]


def run_packet_prediction_replay(
    session: ReplaySession,
    entries: Iterable[ReplayPacketEntry],
    *,
    pipeline: PacketPredictionPipeline,
    control: ReplayControl,
    handle_event: RuntimeEventHandler,
    clock: ReplayClock | None = None,
    poll_interval_seconds: float = 0.1,
    handle_session: RuntimeSessionHandler | None = None,
) -> ReplaySession:
    """Drive packet prediction through controlled replay lifecycle semantics."""

    def handle_entry(entry: ReplayPacketEntry) -> None:
        for event in pipeline.push(entry.packet):
            handle_event(event)

    result = run_controlled_replay_schedule(
        session,
        entries,
        handle_entry=handle_entry,
        control=control,
        clock=clock,
        poll_interval_seconds=poll_interval_seconds,
        handle_session=handle_session,
    )

    if result.state is ReplayState.COMPLETED:
        for event in pipeline.finish():
            handle_event(event)

    return result
