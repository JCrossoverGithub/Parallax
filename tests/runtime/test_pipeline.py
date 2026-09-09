from dataclasses import dataclass
from typing import ClassVar, cast

import pytest

from parallax.data import (
    FlowPacketAssignment,
    PacketMetadata,
    RuntimeObservationWindow,
    WindowExtractionConfig,
)
from parallax.features import RuntimeWindowFeature
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.runtime.pipeline import (
    PacketPredictionPipeline,
    RuntimePipelineError,
)


def _packet() -> PacketMetadata:
    return cast(PacketMetadata, object())


def _assignment() -> FlowPacketAssignment:
    return cast(FlowPacketAssignment, object())


@dataclass(frozen=True, slots=True)
class NamedWindow:
    name: str


@dataclass(frozen=True, slots=True)
class NamedFeature:
    name: str


@dataclass(frozen=True, slots=True)
class NamedPrediction:
    name: str


def _window(name: str) -> RuntimeObservationWindow:
    return cast(RuntimeObservationWindow, NamedWindow(name))


def _feature(name: str) -> RuntimeWindowFeature:
    return cast(RuntimeWindowFeature, NamedFeature(name))


def _prediction(name: str) -> PrototypeRuntimePrediction:
    return cast(PrototypeRuntimePrediction, NamedPrediction(name))


class FakeFlowTracker:
    def __init__(self) -> None:
        self.packets: list[PacketMetadata] = []
        self.assignment = _assignment()

    def push(self, packet: PacketMetadata) -> FlowPacketAssignment:
        self.packets.append(packet)
        return self.assignment


class FakeWindowTracker:
    instances: ClassVar[list["FakeWindowTracker"]] = []

    def __init__(
        self,
        capture_name: str,
        *,
        config: WindowExtractionConfig | None = None,
    ) -> None:
        self.capture_name = capture_name
        self.config = config
        self.assignments: list[FlowPacketAssignment] = []
        self.finish_calls = 0
        FakeWindowTracker.instances.append(self)

    def push(
        self,
        assignment: FlowPacketAssignment,
    ) -> tuple[RuntimeObservationWindow, ...]:
        self.assignments.append(assignment)
        return (_window("first"), _window("second"))

    def finish(self) -> tuple[RuntimeObservationWindow, ...]:
        self.finish_calls += 1
        return (_window("final"),)


class FakeScorer:
    def __init__(self) -> None:
        self.features: list[RuntimeWindowFeature] = []

    def score_feature(
        self,
        feature: RuntimeWindowFeature,
    ) -> PrototypeRuntimePrediction:
        self.features.append(feature)
        return _prediction(cast(NamedFeature, feature).name)


def test_pipeline_composes_packet_through_prediction_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow_tracker = FakeFlowTracker()
    scorer = FakeScorer()

    monkeypatch.setattr(
        "parallax.runtime.pipeline.BidirectionalFlowTracker",
        lambda: flow_tracker,
    )
    monkeypatch.setattr(
        "parallax.runtime.pipeline.IncrementalWindowTracker",
        FakeWindowTracker,
    )
    monkeypatch.setattr(
        "parallax.runtime.pipeline.calculate_runtime_window_feature",
        lambda window: _feature(cast(NamedWindow, window).name),
    )

    config = WindowExtractionConfig()
    pipeline = PacketPredictionPipeline(
        run_id="replay-001",
        capture_name="nonvpn_ssh_capture4.pcap",
        scorer=scorer,
        window_config=config,
    )

    packet = _packet()
    events = pipeline.push(packet)

    assert flow_tracker.packets == [packet]
    assert len(FakeWindowTracker.instances) == 1

    tracker = FakeWindowTracker.instances[0]
    assert tracker.capture_name == "nonvpn_ssh_capture4.pcap"
    assert tracker.config is config
    assert tracker.assignments == [flow_tracker.assignment]

    assert [cast(NamedFeature, feature).name for feature in scorer.features] == [
        "first",
        "second",
    ]
    assert [event.run_id for event in events] == [
        "replay-001",
        "replay-001",
    ]
    assert [cast(NamedPrediction, event.prediction).name for event in events] == ["first", "second"]


def test_finish_flushes_once_and_closes_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = FakeScorer()

    monkeypatch.setattr(
        "parallax.runtime.pipeline.BidirectionalFlowTracker",
        FakeFlowTracker,
    )
    monkeypatch.setattr(
        "parallax.runtime.pipeline.IncrementalWindowTracker",
        FakeWindowTracker,
    )
    monkeypatch.setattr(
        "parallax.runtime.pipeline.calculate_runtime_window_feature",
        lambda window: _feature(cast(NamedWindow, window).name),
    )

    pipeline = PacketPredictionPipeline(
        run_id="replay-002",
        capture_name="nonvpn_ssh_capture4.pcap",
        scorer=scorer,
    )

    events = pipeline.finish()

    assert len(events) == 1
    assert cast(NamedPrediction, events[0].prediction).name == "final"
    assert FakeWindowTracker.instances[-1].finish_calls == 1

    assert pipeline.finish() == ()
    assert FakeWindowTracker.instances[-1].finish_calls == 1

    with pytest.raises(RuntimePipelineError, match="after runtime pipeline is finished"):
        pipeline.push(_packet())


def test_rejects_empty_run_id() -> None:
    with pytest.raises(RuntimePipelineError, match="run ID must not be empty"):
        PacketPredictionPipeline(
            run_id="",
            capture_name="nonvpn_ssh_capture4.pcap",
            scorer=FakeScorer(),
        )
