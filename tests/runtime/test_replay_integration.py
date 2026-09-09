from collections.abc import Callable, Sequence
from pathlib import Path
from socket import inet_aton

import dpkt  # type: ignore[import-untyped]

from parallax.data import (
    IP_PROTOCOL_TCP,
    PCAP_LINKTYPE_RAW_IP,
    WindowExtractionConfig,
    WindowThresholdPolicy,
)
from parallax.features import RuntimeWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.replay import (
    ReplayConfiguration,
    ReplayControl,
    ReplaySession,
    ReplaySessionId,
    ReplayState,
    iter_pcap_replay_entries,
)
from parallax.runtime import (
    PacketPredictionPipeline,
    RuntimePredictionEvent,
    run_packet_prediction_replay,
)

SESSION_ID = "4d524cee-1288-45d7-9c6f-fdbab196bb26"
SOURCE_SHA256 = "a" * 64

SOURCE = "10.101.1.100"
DESTINATION = "10.103.1.100"


class ScriptedClock:
    def __init__(
        self,
        *,
        now: float = 50.0,
        on_sleep: Callable[[int], None] | None = None,
    ) -> None:
        self.now = now
        self.sleep_calls: list[float] = []
        self._on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds

        if self._on_sleep is not None:
            self._on_sleep(len(self.sleep_calls))


class FakeScorer:
    def score_feature(
        self,
        feature: RuntimeWindowFeature,
    ) -> PrototypeRuntimePrediction:
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


def _ipv4_packet() -> bytes:
    transport = dpkt.tcp.TCP(
        sport=41_898,
        dport=22,
        flags=dpkt.tcp.TH_ACK,
        data=b"encrypted",
    )
    packet = dpkt.ip.IP(
        src=inet_aton(SOURCE),
        dst=inet_aton(DESTINATION),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _write_pcap(
    path: Path,
    records: Sequence[tuple[float, bytes]],
) -> None:
    with path.open("wb") as stream:
        writer = dpkt.pcap.Writer(
            stream,
            linktype=PCAP_LINKTYPE_RAW_IP,
        )

        for timestamp, packet in records:
            writer.writepkt(packet, ts=timestamp)

        writer.close()


def _configuration() -> ReplayConfiguration:
    return ReplayConfiguration(time_scale=1.0)


def _session(
    source: Path,
    configuration: ReplayConfiguration,
) -> ReplaySession:
    return ReplaySession(
        session_id=ReplaySessionId.parse(SESSION_ID),
        source_id=source.name,
        source_sha256=SOURCE_SHA256,
        configuration=configuration,
    )


def _pipeline(source: Path) -> PacketPredictionPipeline:
    return PacketPredictionPipeline(
        run_id="replay-001",
        capture_id=source.name,
        scorer=FakeScorer(),
        window_config=WindowExtractionConfig(
            window_seconds=1.0,
            minimum_packets=1,
            threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
        ),
    )


def test_completed_replay_emits_incremental_and_final_predictions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture91.pcap"
    _write_pcap(
        source,
        [
            (100.0, _ipv4_packet()),
            (101.0, _ipv4_packet()),
        ],
    )

    configuration = _configuration()
    events: list[RuntimePredictionEvent] = []

    result = run_packet_prediction_replay(
        _session(source, configuration),
        iter_pcap_replay_entries(
            source,
            configuration=configuration,
        ),
        pipeline=_pipeline(source),
        control=ReplayControl(),
        handle_event=events.append,
        clock=ScriptedClock(),
        poll_interval_seconds=0.25,
    )

    assert result.state is ReplayState.COMPLETED
    assert [event.prediction.window_index for event in events] == [0, 1]
    assert [event.run_id for event in events] == [
        "replay-001",
        "replay-001",
    ]


def test_pause_resume_preserves_prediction_replay_completion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture92.pcap"
    _write_pcap(
        source,
        [
            (100.0, _ipv4_packet()),
            (103.0, _ipv4_packet()),
        ],
    )

    configuration = _configuration()
    control = ReplayControl()
    events: list[RuntimePredictionEvent] = []
    states: list[ReplayState] = []

    def on_sleep(call_number: int) -> None:
        if call_number == 1:
            control.pause()
        elif call_number == 3:
            control.resume()

    result = run_packet_prediction_replay(
        _session(source, configuration),
        iter_pcap_replay_entries(
            source,
            configuration=configuration,
        ),
        pipeline=_pipeline(source),
        control=control,
        handle_event=events.append,
        clock=ScriptedClock(on_sleep=on_sleep),
        poll_interval_seconds=1.0,
        handle_session=lambda replay: states.append(replay.state),
    )

    assert result.state is ReplayState.COMPLETED
    assert states == [
        ReplayState.RUNNING,
        ReplayState.PAUSED,
        ReplayState.RUNNING,
        ReplayState.COMPLETED,
    ]
    assert [event.prediction.window_index for event in events] == [0, 3]


def test_cancelled_replay_does_not_flush_incomplete_pipeline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture93.pcap"
    _write_pcap(
        source,
        [
            (100.0, _ipv4_packet()),
            (400.0, _ipv4_packet()),
        ],
    )

    configuration = _configuration()
    control = ReplayControl()
    events: list[RuntimePredictionEvent] = []

    def on_sleep(_: int) -> None:
        control.cancel()

    result = run_packet_prediction_replay(
        _session(source, configuration),
        iter_pcap_replay_entries(
            source,
            configuration=configuration,
        ),
        pipeline=_pipeline(source),
        control=control,
        handle_event=events.append,
        clock=ScriptedClock(on_sleep=on_sleep),
        poll_interval_seconds=0.25,
    )

    assert result.state is ReplayState.CANCELLED

    # The first packet remains buffered. Cancellation must not turn that
    # incomplete replay state into a final prediction.
    assert events == []
