"""Live packet-source integration for the prediction runtime."""

import time
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
    """Operational summary of one bounded live inference run."""

    packets_processed: int
    events_emitted: int
    elapsed_seconds: float
    packet_rate_per_second: float
    event_rate_per_second: float
    mean_processing_latency_ms: float
    max_processing_latency_ms: float


def run_live_packet_predictions(
    source: RuntimePacketSource,
    *,
    pipeline: RuntimePacketPipeline,
    packet_limit: int,
    handle_event: LiveRuntimeEventHandler,
    clock: Callable[[], float] | None = None,
) -> LiveRuntimeSummary:
    """Run a bounded live packet source through the shared prediction pipeline."""
    if packet_limit < 1:
        raise LiveRuntimeError("live packet limit must be positive")

    selected_clock = clock if clock is not None else time.perf_counter

    packets_processed = 0
    events_emitted = 0
    total_processing_seconds = 0.0
    max_processing_seconds = 0.0

    source.open()
    started_at = selected_clock()

    try:
        for _ in range(packet_limit):
            packet = source.receive()
            packets_processed += 1

            processing_started_at = selected_clock()
            events = pipeline.push(packet)
            processing_seconds = selected_clock() - processing_started_at

            total_processing_seconds += processing_seconds
            max_processing_seconds = max(
                max_processing_seconds,
                processing_seconds,
            )

            for event in events:
                handle_event(event)
                events_emitted += 1

        for event in pipeline.finish():
            handle_event(event)
            events_emitted += 1

        elapsed_seconds = selected_clock() - started_at
    finally:
        source.close()

    if elapsed_seconds > 0.0:
        packet_rate_per_second = packets_processed / elapsed_seconds
        event_rate_per_second = events_emitted / elapsed_seconds
    else:
        packet_rate_per_second = 0.0
        event_rate_per_second = 0.0

    mean_processing_latency_ms = (total_processing_seconds / packets_processed) * 1_000.0

    return LiveRuntimeSummary(
        packets_processed=packets_processed,
        events_emitted=events_emitted,
        elapsed_seconds=elapsed_seconds,
        packet_rate_per_second=packet_rate_per_second,
        event_rate_per_second=event_rate_per_second,
        mean_processing_latency_ms=mean_processing_latency_ms,
        max_processing_latency_ms=max_processing_seconds * 1_000.0,
    )
