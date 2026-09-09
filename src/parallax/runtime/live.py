"""Live packet-source integration for the prediction runtime."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from parallax.data import PacketMetadata
from parallax.runtime.events import RuntimePredictionEvent


class LiveRuntimeError(ValueError):
    """Raised when live runtime execution parameters are invalid."""


class RuntimePacketSource(Protocol):
    """Structural packet-source contract required by live inference."""

    def open(self) -> None:
        """Open the underlying packet source."""

    def receive(self) -> PacketMetadata:
        """Return the next supported packet."""

    def close(self) -> None:
        """Close the underlying packet source."""


class RuntimePacketPipeline(Protocol):
    """Structural prediction-pipeline contract required by live inference."""

    def push(
        self,
        packet: PacketMetadata,
    ) -> tuple[RuntimePredictionEvent, ...]:
        """Consume one packet and return newly completed predictions."""

    def finish(self) -> tuple[RuntimePredictionEvent, ...]:
        """Flush final eligible windows."""


LiveRuntimeEventHandler = Callable[[RuntimePredictionEvent], None]


@dataclass(frozen=True, slots=True)
class LiveRuntimeSummary:
    """Summary of one bounded live inference run."""

    packets_processed: int
    events_emitted: int


def run_live_packet_predictions(
    source: RuntimePacketSource,
    *,
    pipeline: RuntimePacketPipeline,
    packet_limit: int,
    handle_event: LiveRuntimeEventHandler,
) -> LiveRuntimeSummary:
    """Run a bounded live packet source through the shared prediction pipeline."""
    if packet_limit < 1:
        raise LiveRuntimeError("live packet limit must be positive")

    packets_processed = 0
    events_emitted = 0

    source.open()

    try:
        for _ in range(packet_limit):
            packet = source.receive()
            packets_processed += 1

            for event in pipeline.push(packet):
                handle_event(event)
                events_emitted += 1

        for event in pipeline.finish():
            handle_event(event)
            events_emitted += 1
    finally:
        source.close()

    return LiveRuntimeSummary(
        packets_processed=packets_processed,
        events_emitted=events_emitted,
    )
