from collections import deque
from socket import inet_aton

import dpkt  # type: ignore[import-untyped]

from parallax.data import (
    IP_PROTOCOL_TCP,
    WindowExtractionConfig,
)
from parallax.features import RuntimeWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.runtime import (
    PacketPredictionPipeline,
    RuntimePredictionEvent,
    run_live_packet_predictions,
)
from parallax.sensor import (
    CaptureInterface,
    LiveEthernetCapture,
    LivePacketSource,
)

_SOURCE = "10.0.0.10"
_DESTINATION = "10.0.0.20"

_DESTINATION_MAC = bytes.fromhex("001122334455")
_SOURCE_MAC = bytes.fromhex("66778899aabb")


class QueueSocket:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = deque(frames)
        self.closed = False

    def bind(self, address: tuple[str, int]) -> None:
        assert address == ("eth0", 0)

    def recv(self, bufsize: int) -> bytes:
        assert bufsize == 65_535

        if not self.frames:
            raise AssertionError("test Ethernet frame queue exhausted")

        return self.frames.popleft()

    def close(self) -> None:
        self.closed = True


class ScriptedClock:
    def __init__(self, timestamps: list[float]) -> None:
        self.timestamps = deque(timestamps)

    def __call__(self) -> float:
        if not self.timestamps:
            raise AssertionError("test timestamp queue exhausted")

        return self.timestamps.popleft()


class FakeScorer:
    def __init__(self) -> None:
        self.features: list[RuntimeWindowFeature] = []

    def score_feature(
        self,
        feature: RuntimeWindowFeature,
    ) -> PrototypeRuntimePrediction:
        self.features.append(feature)

        probabilities = (0.7, 0.1, 0.1, 0.05, 0.05)

        return PrototypeRuntimePrediction(
            window_id=feature.window_id,
            capture_id=feature.capture_id,
            flow_id=feature.flow_id,
            window_index=feature.window_index,
            start_offset_seconds=feature.start_offset_seconds,
            end_offset_seconds=feature.end_offset_seconds,
            packet_count=feature.packet_count,
            category_order=CATEGORY_LABELS,
            class_probabilities=probabilities,
            predicted_class_index=0,
            predicted_category=CATEGORY_LABELS[0],
            raw_confidence=probabilities[0],
            relative_mahalanobis_distance=1.0,
            ood_score=0.1,
            model_bundle_sha256="b" * 64,
            calibration_artifact_sha256="c" * 64,
            feature_artifact_sha256="d" * 64,
            split_manifest_sha256="e" * 64,
        )


def _ethernet_tcp_frame() -> bytes:
    transport = dpkt.tcp.TCP(
        sport=50_000,
        dport=443,
        flags=dpkt.tcp.TH_ACK,
        data=b"encrypted",
    )

    packet = dpkt.ip.IP(
        src=inet_aton(_SOURCE),
        dst=inet_aton(_DESTINATION),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)

    return _DESTINATION_MAC + _SOURCE_MAC + b"\x08\x00" + bytes(packet)


def test_live_ethernet_frames_reach_shared_prediction_pipeline() -> None:
    frame = _ethernet_tcp_frame()
    fake_socket = QueueSocket(
        [
            frame,
            frame,
            frame,
            frame,
        ]
    )

    capture = LiveEthernetCapture(
        CaptureInterface(index=2, name="eth0"),
        socket_factory=lambda: fake_socket,
        clock=ScriptedClock(
            [
                100.0,
                100.2,
                100.4,
                101.2,
            ]
        ),
    )
    source = LivePacketSource(capture)

    scorer = FakeScorer()
    capture_id = "live:eth0:session-001"

    pipeline = PacketPredictionPipeline(
        run_id="live-run-001",
        capture_id=capture_id,
        scorer=scorer,
        window_config=WindowExtractionConfig(
            window_seconds=1.0,
            minimum_packets=1,
        ),
    )

    events: list[RuntimePredictionEvent] = []

    summary = run_live_packet_predictions(
        source,
        pipeline=pipeline,
        packet_limit=4,
        handle_event=events.append,
    )

    assert summary.packets_processed == 4
    assert summary.events_emitted == 1

    assert fake_socket.closed

    assert len(scorer.features) == 1
    assert scorer.features[0].capture_id == capture_id
    assert scorer.features[0].packet_count == 3

    assert len(events) == 1
    assert events[0].run_id == "live-run-001"
    assert events[0].prediction.capture_id == capture_id
    assert events[0].prediction.packet_count == 3
