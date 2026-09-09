from collections import deque
from pathlib import Path
from threading import Event

import pytest

import parallax.operator.live_runtime as live_runtime
from parallax.data import (
    FEATURE_COUNT,
    IP_PROTOCOL_TCP,
    PacketMetadata,
)
from parallax.features import RuntimeWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.operator import (
    OperatorLiveConfiguration,
    OperatorLiveRuntimeExecutor,
)
from parallax.runtime import RuntimePredictionEvent
from parallax.runtime.live import RuntimePacketSource


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
            start_offset_seconds=(feature.start_offset_seconds),
            end_offset_seconds=(feature.end_offset_seconds),
            packet_count=feature.packet_count,
            category_order=CATEGORY_LABELS,
            class_probabilities=probabilities,
            predicted_class_index=0,
            predicted_category=(CATEGORY_LABELS[0]),
            raw_confidence=probabilities[0],
            relative_mahalanobis_distance=1.0,
            ood_score=0.1,
            model_bundle_sha256="a" * 64,
            calibration_artifact_sha256="b" * 64,
            feature_artifact_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
        )


class StoppingPacketSource:
    def __init__(
        self,
        packets: list[PacketMetadata],
        stop_event: Event,
    ) -> None:
        self.packets = deque(packets)
        self.stop_event = stop_event
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def receive(
        self,
    ) -> PacketMetadata | None:
        packet = self.packets.popleft()

        if not self.packets:
            self.stop_event.set()

        return packet

    def close(self) -> None:
        self.closed = True


def _packets() -> list[PacketMetadata]:
    return [
        PacketMetadata(
            timestamp_seconds=(100.0 + index * 0.01),
            source_address="10.0.0.10",
            source_port=41_898,
            destination_address="10.0.0.20",
            destination_port=443,
            protocol=IP_PROTOCOL_TCP,
            size=100,
        )
        for index in range(21)
    ]


def test_operator_executor_runs_ipc_packet_pipeline() -> None:
    run_id = "live-run-001"
    stop_event = Event()
    source = StoppingPacketSource(
        _packets(),
        stop_event,
    )
    scorer = FakeScorer()

    factory_calls: list[tuple[Path, str]] = []

    def make_source(
        socket_path: Path,
        interface: str,
    ) -> StoppingPacketSource:
        factory_calls.append(
            (
                socket_path,
                interface,
            )
        )
        return source

    executor = OperatorLiveRuntimeExecutor(
        scorer,
        sensor_socket_path=("/tmp/parallax-sensor.sock"),
        source_factory=make_source,
    )

    events: list[RuntimePredictionEvent] = []

    summary = executor(
        run_id,
        OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
        ),
        stop_event,
        events.append,
    )

    assert factory_calls == [
        (
            Path("/tmp/parallax-sensor.sock"),
            "eth0",
        )
    ]

    assert source.opened
    assert source.closed

    assert summary.packets_processed == 21
    assert summary.events_emitted == 1

    assert scorer.capture_ids == ["live:eth0:live-run-001"]

    assert len(events) == 1


def test_default_factory_uses_sensor_ipc_packet_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "live-run-default"
    stop_event = Event()
    source = StoppingPacketSource(
        _packets(),
        stop_event,
    )
    scorer = FakeScorer()

    calls: list[tuple[Path, str]] = []

    def fake_ipc_source(
        socket_path: str | Path,
        interface: str,
    ) -> StoppingPacketSource:
        calls.append(
            (
                Path(socket_path),
                interface,
            )
        )
        return source

    monkeypatch.setattr(
        live_runtime,
        "SensorIpcPacketSource",
        fake_ipc_source,
    )

    executor = OperatorLiveRuntimeExecutor(
        scorer,
        sensor_socket_path=("/tmp/default-sensor.sock"),
    )

    executor(
        run_id,
        OperatorLiveConfiguration(
            interface="eth0",
            stale_after_seconds=120.0,
            max_tracked_flows=4_096,
        ),
        stop_event,
        lambda event: None,
    )

    assert calls == [
        (
            Path("/tmp/default-sensor.sock"),
            "eth0",
        )
    ]


def test_operator_executor_preserves_sensor_failure_code() -> None:
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )
    from parallax.sensor.ipc_client import (
        SensorIpcClientError,
    )

    def fail_source(
        socket_path: Path,
        interface: str,
    ) -> RuntimePacketSource:
        assert socket_path == Path("/tmp/parallax-sensor.sock")
        assert interface == "eth0"

        raise SensorIpcClientError(
            "permission denied",
            code="capture_error",
        )

    executor = OperatorLiveRuntimeExecutor(
        FakeScorer(),
        sensor_socket_path=("/tmp/parallax-sensor.sock"),
        source_factory=fail_source,
    )

    with pytest.raises(
        OperatorLiveExecutionError,
        match="permission denied",
    ) as failure:
        executor(
            "live-structured-error",
            OperatorLiveConfiguration(
                interface="eth0",
                stale_after_seconds=120.0,
                max_tracked_flows=4_096,
            ),
            Event(),
            lambda event: None,
        )

    assert failure.value.code == "capture_error"


def test_operator_executor_classifies_unstructured_ipc_failure() -> None:
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )
    from parallax.sensor.ipc_client import (
        SensorIpcClientError,
    )

    def fail_source(
        socket_path: Path,
        interface: str,
    ) -> RuntimePacketSource:
        raise SensorIpcClientError("could not open sensor IPC packet source")

    executor = OperatorLiveRuntimeExecutor(
        FakeScorer(),
        source_factory=fail_source,
    )

    with pytest.raises(
        OperatorLiveExecutionError,
        match=("could not open sensor IPC packet source"),
    ) as failure:
        executor(
            "live-ipc-error",
            OperatorLiveConfiguration(
                interface="eth0",
                stale_after_seconds=120.0,
                max_tracked_flows=4_096,
            ),
            Event(),
            lambda event: None,
        )

    assert failure.value.code == "sensor_ipc_error"


def test_operator_executor_reports_flow_capacity_exceeded() -> None:
    from parallax.operator.live import (
        OperatorLiveExecutionError,
    )

    stop_event = Event()

    source = StoppingPacketSource(
        [
            PacketMetadata(
                timestamp_seconds=1.0,
                source_address="10.0.0.1",
                source_port=50_000,
                destination_address="10.0.0.2",
                destination_port=443,
                protocol=IP_PROTOCOL_TCP,
                size=100,
            ),
            PacketMetadata(
                timestamp_seconds=2.0,
                source_address="10.0.0.1",
                source_port=50_001,
                destination_address="10.0.0.2",
                destination_port=443,
                protocol=IP_PROTOCOL_TCP,
                size=100,
            ),
        ],
        stop_event,
    )

    executor = OperatorLiveRuntimeExecutor(
        FakeScorer(),
        source_factory=lambda socket_path, interface: source,
    )

    with pytest.raises(
        OperatorLiveExecutionError,
        match="runtime flow capacity reached: 1",
    ) as failure:
        executor(
            "live-capacity",
            OperatorLiveConfiguration(
                interface="eth0",
                stale_after_seconds=120.0,
                max_tracked_flows=1,
            ),
            stop_event,
            lambda event: None,
        )

    assert failure.value.code == "flow_capacity_exceeded"
    assert source.opened
    assert source.closed
