from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from socket import inet_aton

import dpkt  # type: ignore[import-untyped]
import numpy as np

from parallax.data import (
    FEATURE_COUNT,
    PCAP_LINKTYPE_RAW_IP,
    PacketSizePolicy,
    WindowExtractionConfig,
    WindowThresholdPolicy,
)
from parallax.features import (
    ByteTotalPolicy,
    FeatureCalculationConfig,
    extract_vnat_pcap_features,
)

GOLDEN_VECTOR_SHA256 = "6d2292590e4d741c9c230d099932e24bd2fe470cbb8a8c7e14edbe973b2c1420"


def _tcp_packet(
    *,
    source: str,
    source_port: int,
    destination: str,
    destination_port: int,
    payload: bytes,
) -> bytes:
    transport = dpkt.tcp.TCP(
        sport=source_port,
        dport=destination_port,
        flags=dpkt.tcp.TH_ACK,
        data=payload,
    )
    packet = dpkt.ip.IP(
        src=inet_aton(source),
        dst=inet_aton(destination),
        p=6,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _udp_packet(*, payload: bytes) -> bytes:
    transport = dpkt.udp.UDP(
        sport=5_000,
        dport=5_001,
        data=payload,
    )
    transport.ulen = len(transport)
    packet = dpkt.ip.IP(
        src=inet_aton("10.0.0.3"),
        dst=inet_aton("10.0.0.4"),
        p=17,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _write_pcap(path: Path, records: Sequence[tuple[float, bytes]]) -> None:
    with path.open("wb") as stream:
        writer = dpkt.pcap.Writer(stream, linktype=PCAP_LINKTYPE_RAW_IP)
        for timestamp, packet in records:
            writer.writepkt(packet, ts=timestamp)
        writer.close()


def test_synthetic_pcap_matches_frozen_golden_feature_vector(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture1.pcap"
    records: list[tuple[float, bytes]] = []

    for index in range(25):
        forward = index % 3 != 1
        records.append(
            (
                1_000.0 + index * index * 0.017,
                _tcp_packet(
                    source="10.0.0.1" if forward else "10.0.0.2",
                    source_port=41_898 if forward else 22,
                    destination="10.0.0.2" if forward else "10.0.0.1",
                    destination_port=22 if forward else 41_898,
                    payload=bytes([index]) * (5 + index * 3),
                ),
            )
        )

    _write_pcap(source, records)

    feature = next(extract_vnat_pcap_features(source))

    assert feature.capture.capture_id == source.name
    assert feature.capture.application.value == "ssh"
    assert feature.capture.category.value == "C2"
    assert feature.window_index == 0
    assert feature.start_offset_seconds == 0.0
    assert feature.end_offset_seconds == 40.96
    assert feature.packet_count == 25
    assert feature.values.shape == (FEATURE_COUNT,)
    assert feature.values.dtype == np.float32
    assert feature.values.flags.writeable is False
    assert sha256(feature.values.tobytes()).hexdigest() == GOLDEN_VECTOR_SHA256
    assert not hasattr(feature, "timestamps")
    assert not hasattr(feature, "sizes")
    assert not hasattr(feature, "directions")


def test_forwards_packet_and_feature_compatibility_policies(tmp_path: Path) -> None:
    source = tmp_path / "vpn_voip_capture1.pcap"
    packet = _udp_packet(payload=b"voice-frame")
    _write_pcap(source, [(float(index), packet) for index in range(21)])

    release = next(extract_vnat_pcap_features(source))
    corrected = next(
        extract_vnat_pcap_features(
            source,
            size_policy=PacketSizePolicy.IP_PACKET,
            feature_config=FeatureCalculationConfig(
                byte_total_policy=ByteTotalPolicy.CORRECTED,
            ),
        )
    )

    assert not np.array_equal(release.values, corrected.values)


def test_forwards_window_configuration(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture1.pcap"
    packet = _tcp_packet(
        source="10.0.0.1",
        source_port=41_898,
        destination="10.0.0.2",
        destination_port=22,
        payload=b"encrypted",
    )
    _write_pcap(source, [(float(index), packet) for index in range(20)])

    release = list(extract_vnat_pcap_features(source))
    literal = list(
        extract_vnat_pcap_features(
            source,
            window_config=WindowExtractionConfig(
                threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
            ),
        )
    )

    assert release == []
    assert len(literal) == 1
    assert literal[0].packet_count == 20


def test_incremental_windows_match_batch_pcap_features(tmp_path: Path) -> None:
    from parallax.data import (
        BidirectionalFlowTracker,
        IncrementalWindowTracker,
        WindowExtractionConfig,
        WindowThresholdPolicy,
        iter_pcap_packet_metadata,
    )
    from parallax.features.runtime import calculate_vnat_window_feature

    source = tmp_path / "nonvpn_ssh_capture91.pcap"

    records: list[tuple[float, bytes]] = []
    for index in range(21):
        records.append(
            (
                100.0 + index * 0.1,
                _tcp_packet(
                    source="10.101.1.100",
                    destination="10.103.1.100",
                    source_port=41_898,
                    destination_port=22,
                    payload=b"encrypted",
                ),
            )
        )

    for index in range(21):
        records.append(
            (
                141.0 + index * 0.1,
                _tcp_packet(
                    source="10.101.1.100",
                    destination="10.103.1.100",
                    source_port=41_898,
                    destination_port=22,
                    payload=b"encrypted",
                ),
            )
        )

    _write_pcap(source, records)

    window_config = WindowExtractionConfig(
        window_seconds=40.96,
        minimum_packets=20,
        threshold_policy=WindowThresholdPolicy.RELEASE_COMPATIBLE,
    )

    expected = list(
        extract_vnat_pcap_features(
            source,
            window_config=window_config,
        )
    )

    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(
        source.name,
        config=window_config,
    )

    actual = []

    for packet in iter_pcap_packet_metadata(source):
        assignment = flow_tracker.push(packet)

        for window in window_tracker.push(assignment):
            actual.append(calculate_vnat_window_feature(window))

    actual.extend(calculate_vnat_window_feature(window) for window in window_tracker.finish())

    assert [feature.window_id for feature in actual] == [feature.window_id for feature in expected]
    assert [feature.packet_count for feature in actual] == [
        feature.packet_count for feature in expected
    ]

    for actual_feature, expected_feature in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            actual_feature.values,
            expected_feature.values,
        )


def test_calculate_vnat_window_feature_preserves_window_provenance(
    tmp_path: Path,
) -> None:
    from parallax.data import (
        BidirectionalFlowTracker,
        IncrementalWindowTracker,
        WindowExtractionConfig,
        WindowThresholdPolicy,
        iter_pcap_packet_metadata,
    )
    from parallax.features.runtime import calculate_vnat_window_feature

    source = tmp_path / "nonvpn_ssh_capture92.pcap"
    _write_pcap(
        source,
        [
            (
                100.0 + index * 0.1,
                _tcp_packet(
                    source="10.101.1.100",
                    destination="10.103.1.100",
                    source_port=41_898,
                    destination_port=22,
                    payload=b"encrypted",
                ),
            )
            for index in range(21)
        ],
    )

    config = WindowExtractionConfig(
        minimum_packets=20,
        threshold_policy=WindowThresholdPolicy.RELEASE_COMPATIBLE,
    )
    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(source.name, config=config)

    for packet in iter_pcap_packet_metadata(source):
        assert window_tracker.push(flow_tracker.push(packet)) == ()

    window = window_tracker.finish()[0]
    feature = calculate_vnat_window_feature(window)

    assert feature.window_id == window.window_id
    assert feature.capture == window.capture
    assert feature.flow_id == window.flow_id
    assert feature.window_index == window.window_index
    assert feature.start_offset_seconds == window.start_offset_seconds
    assert feature.end_offset_seconds == window.end_offset_seconds
    assert feature.packet_count == window.packet_count
