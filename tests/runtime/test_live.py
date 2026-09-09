from collections import deque
from dataclasses import dataclass
from typing import cast

import pytest

from parallax.data import PacketMetadata
from parallax.runtime import (
    LiveRuntimeError,
    LiveRuntimeSummary,
    RuntimePredictionEvent,
    run_live_packet_predictions,
)


@dataclass(frozen=True, slots=True)
class NamedPacket:
    name: str


@dataclass(frozen=True, slots=True)
class NamedEvent:
    name: str


def _packet(name: str) -> PacketMetadata:
    return cast(PacketMetadata, NamedPacket(name))


def _event(name: str) -> RuntimePredictionEvent:
    return cast(RuntimePredictionEvent, NamedEvent(name))


class FakeSource:
    def __init__(
        self,
        packets: list[PacketMetadata],
        *,
        receive_error: Exception | None = None,
    ) -> None:
        self.packets = deque(packets)
        self.receive_error = receive_error
        self.open_calls = 0
        self.close_calls = 0
        self.receive_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def receive(self) -> PacketMetadata | None:
        self.receive_calls += 1

        if self.receive_error is not None:
            raise self.receive_error

        if not self.packets:
            raise AssertionError("test packet source exhausted")

        return self.packets.popleft()

    def close(self) -> None:
        self.close_calls += 1


class FakePipeline:
    def __init__(
        self,
        push_events: list[tuple[RuntimePredictionEvent, ...]],
        *,
        finish_events: tuple[RuntimePredictionEvent, ...] = (),
    ) -> None:
        self.push_events = deque(push_events)
        self.finish_events = finish_events
        self.packets: list[PacketMetadata] = []
        self.finish_calls = 0

    def push(
        self,
        packet: PacketMetadata,
    ) -> tuple[RuntimePredictionEvent, ...]:
        self.packets.append(packet)

        if not self.push_events:
            return ()

        return self.push_events.popleft()

    def finish(self) -> tuple[RuntimePredictionEvent, ...]:
        self.finish_calls += 1
        return self.finish_events


class ScriptedClock:
    def __init__(self, timestamps: list[float]) -> None:
        self.timestamps = deque(timestamps)

    def __call__(self) -> float:
        if not self.timestamps:
            raise AssertionError("test clock exhausted")

        return self.timestamps.popleft()


@pytest.mark.parametrize("packet_limit", [0, -1])
def test_rejects_nonpositive_packet_limit(packet_limit: int) -> None:
    source = FakeSource([])

    with pytest.raises(
        LiveRuntimeError,
        match="live packet limit must be positive",
    ):
        run_live_packet_predictions(
            source,
            pipeline=FakePipeline([]),
            packet_limit=packet_limit,
            handle_event=lambda event: None,
        )

    assert source.open_calls == 0
    assert source.close_calls == 0


def test_drives_packets_events_and_final_flush() -> None:
    first_packet = _packet("first")
    second_packet = _packet("second")

    first_event = _event("first-event")
    final_event = _event("final-event")

    source = FakeSource(
        [
            first_packet,
            second_packet,
        ]
    )
    pipeline = FakePipeline(
        [
            (first_event,),
            (),
        ],
        finish_events=(final_event,),
    )
    events: list[RuntimePredictionEvent] = []

    summary = run_live_packet_predictions(
        source,
        pipeline=pipeline,
        packet_limit=2,
        handle_event=events.append,
        clock=ScriptedClock(
            [
                10.0,
                10.125,
                10.25,
                10.375,
                10.625,
                10.625,
                10.875,
                11.0,
            ]
        ),
    )

    assert summary == LiveRuntimeSummary(
        packets_processed=2,
        events_emitted=2,
        elapsed_seconds=1.0,
        packet_rate_per_second=2.0,
        event_rate_per_second=2.0,
        mean_processing_latency_ms=187.5,
        max_processing_latency_ms=250.0,
        finalization_latency_ms=250.0,
        total_pipeline_processing_ms=625.0,
    )
    assert source.open_calls == 1
    assert source.receive_calls == 2
    assert source.close_calls == 1
    assert pipeline.packets == [
        first_packet,
        second_packet,
    ]
    assert pipeline.finish_calls == 1
    assert events == [
        first_event,
        final_event,
    ]


def test_source_is_closed_when_receive_fails() -> None:
    source = FakeSource(
        [],
        receive_error=RuntimeError("capture failed"),
    )
    pipeline = FakePipeline([])

    with pytest.raises(
        RuntimeError,
        match="capture failed",
    ):
        run_live_packet_predictions(
            source,
            pipeline=pipeline,
            packet_limit=1,
            handle_event=lambda event: None,
        )

    assert source.open_calls == 1
    assert source.close_calls == 1
    assert pipeline.finish_calls == 0


def test_zero_elapsed_time_reports_zero_rates() -> None:
    source = FakeSource([_packet("only")])
    pipeline = FakePipeline([()])

    summary = run_live_packet_predictions(
        source,
        pipeline=pipeline,
        packet_limit=1,
        handle_event=lambda event: None,
        clock=ScriptedClock(
            [
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ]
        ),
    )

    assert summary == LiveRuntimeSummary(
        packets_processed=1,
        events_emitted=0,
        elapsed_seconds=0.0,
        packet_rate_per_second=0.0,
        event_rate_per_second=0.0,
        mean_processing_latency_ms=0.0,
        max_processing_latency_ms=0.0,
        finalization_latency_ms=0.0,
        total_pipeline_processing_ms=0.0,
    )


def test_requires_packet_limit_or_stop_signal() -> None:
    source = FakeSource([])

    with pytest.raises(
        LiveRuntimeError,
        match="requires a packet limit or stop signal",
    ):
        run_live_packet_predictions(
            source,
            pipeline=FakePipeline([]),
            handle_event=lambda event: None,
        )

    assert source.open_calls == 0


def test_empty_polls_do_not_count_as_packets() -> None:
    class PollingSource(FakeSource):
        def __init__(self) -> None:
            super().__init__([_packet("only")])
            self.empty_polls = 2

        def receive(self) -> PacketMetadata | None:
            self.receive_calls += 1

            if self.empty_polls > 0:
                self.empty_polls -= 1
                return None

            return self.packets.popleft()

    source = PollingSource()
    pipeline = FakePipeline([()])

    summary = run_live_packet_predictions(
        source,
        pipeline=pipeline,
        packet_limit=1,
        handle_event=lambda event: None,
        clock=ScriptedClock(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        ),
    )

    assert source.receive_calls == 3
    assert summary.packets_processed == 1


def test_stop_signal_ends_unbounded_run_and_finalizes() -> None:
    source = FakeSource([])
    pipeline = FakePipeline(
        [],
        finish_events=(_event("final"),),
    )
    events: list[RuntimePredictionEvent] = []
    stop_checks = 0

    def stop_requested() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks >= 1

    summary = run_live_packet_predictions(
        source,
        pipeline=pipeline,
        stop_requested=stop_requested,
        handle_event=events.append,
        clock=ScriptedClock(
            [
                10.0,
                10.0,
                10.0,
                10.0,
            ]
        ),
    )

    assert summary.packets_processed == 0
    assert summary.events_emitted == 1
    assert summary.mean_processing_latency_ms == 0.0
    assert source.receive_calls == 0
    assert source.close_calls == 1
    assert pipeline.finish_calls == 1
    assert events == [_event("final")]
