from collections import deque
from socket import inet_aton
from threading import Event

import dpkt  # type: ignore[import-untyped]

from parallax.data import (
    FEATURE_COUNT,
    IP_PROTOCOL_TCP,
)
from parallax.features import RuntimeWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.operator import (
    OperatorLiveConfiguration,
    OperatorLiveRuntimeExecutor,
)
from parallax.sensor import (
    CaptureInterface,
    LiveEthernetCapture,
)

_DESTINATION_MAC = bytes.fromhex("001122334455")
_SOURCE_MAC = bytes.fromhex("66778899aabb")


class FakeScorer:
    def __init__(self) -> None:
        self.capture_ids: list[str] = []

    def score_feature(
        self,
        feature: RuntimeWindowFeature,
    ) -> PrototypeRuntimePrediction:
        assert feature.values.shape == (FEATURE_COUNT,)
        self.capture_ids.append(feature.capture_id)

        probabilities = (
            0.7,
            0.1,
            0.1,
            0.05,
            0.05,
        )

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
            model_bundle_sha256="a" * 64,
            calibration_artifact_sha256="b" * 64,
            feature_artifact_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
        )


class StoppingSocket:
    def __init__(
        self,
        frames: list[bytes],
        stop_event: Event,
    ) -> None:
        self.frames = deque(frames)
        self.stop_event = stop_event
        self.bound_address: tuple[str, int] | None = None
        self.timeout_seconds: float | None = None
        self.closed = False

    def bind(
        self,
        address: tuple[str, int],
    ) -> None:
        self.bound_address = address

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        self.timeout_seconds = value

    def recv(
        self,
        bufsize: int,
    ) -> bytes:
        assert bufsize == 65_535

        frame = self.frames.popleft()

        if not self.frames:
            self.stop_event.set()

        return frame

    def close(self) -> None:
        self.closed = True


def _ipv4_tcp_frame() -> bytes:
    transport = dpkt.tcp.TCP(
        sport=41_898,
        dport=443,
        flags=dpkt.tcp.TH_ACK,
        data=b"opaque",
    )

    packet = dpkt.ip.IP(
        src=inet_aton("10.0.0.10"),
        dst=inet_aton("10.0.0.20"),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)

    return _DESTINATION_MAC + _SOURCE_MAC + b"\x08\x00" + bytes(packet)


def test_operator_executor_runs_real_live_pipeline() -> None:
    run_id = "live-run-001"
    interface = CaptureInterface(
        index=2,
        name="eth0",
    )
    stop_event = Event()
    socket = StoppingSocket(
        [_ipv4_tcp_frame() for _ in range(21)],
        stop_event,
    )
    scorer = FakeScorer()

    def resolve_interface(
        name: str,
    ) -> CaptureInterface:
        assert name == "eth0"
        return interface

    def make_capture(
        selected: CaptureInterface,
    ) -> LiveEthernetCapture:
        assert selected == interface

        timestamps = iter(100.0 + index * 0.01 for index in range(21))

        return LiveEthernetCapture(
            selected,
            socket_factory=lambda: socket,
            clock=lambda: next(timestamps),
        )

    executor = OperatorLiveRuntimeExecutor(
        scorer,
        interface_resolver=resolve_interface,
        capture_factory=make_capture,
    )

    summary = executor(
        run_id,
        OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
        ),
        stop_event,
    )

    assert summary.packets_processed == 21
    assert summary.events_emitted == 1

    assert scorer.capture_ids == ["live:eth0:live-run-001"]

    assert socket.bound_address == ("eth0", 0)
    assert socket.timeout_seconds == 0.25
    assert socket.closed
