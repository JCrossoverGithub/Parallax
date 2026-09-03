from collections.abc import Sequence
from pathlib import Path
from socket import inet_aton

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import (
    PCAP_LINKTYPE_RAW_IP,
    PacketSizePolicy,
    VnatWindowError,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    extract_vnat_pcap_windows,
)


def _packet(
    *,
    source: str = "10.101.1.100",
    source_port: int = 41_898,
    destination: str = "10.103.1.100",
    destination_port: int = 22,
    protocol: int = 6,
    payload: bytes = b"encrypted",
) -> bytes:
    if protocol == 6:
        transport = dpkt.tcp.TCP(
            sport=source_port,
            dport=destination_port,
            flags=dpkt.tcp.TH_ACK,
            data=payload,
        )
    else:
        transport = dpkt.udp.UDP(
            sport=source_port,
            dport=destination_port,
            data=payload,
        )
        transport.ulen = len(transport)

    packet = dpkt.ip.IP(
        src=inet_aton(source),
        dst=inet_aton(destination),
        p=protocol,
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


def test_extracts_vnat_pcap_through_shared_windowing_path(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture1.pcap"
    forward = _packet()
    reverse = _packet(
        source="10.103.1.100",
        source_port=22,
        destination="10.101.1.100",
        destination_port=41_898,
    )
    records = [
        (100.0 + index * 0.01, forward if index % 2 == 0 else reverse) for index in range(21)
    ]
    _write_pcap(source, records)

    windows = list(extract_vnat_pcap_windows(source))

    assert len(windows) == 1
    window = windows[0]
    assert window.capture.capture_id == source.name
    assert window.capture.application.value == "ssh"
    assert window.connection == ("10.101.1.100", 41_898, "10.103.1.100", 22, 6)
    assert window.window_index == 0
    assert window.packet_count == 21
    assert window.timestamps.tolist() == pytest.approx([index * 0.01 for index in range(21)])
    assert window.sizes.tolist() == [len(forward)] * 21
    assert window.directions.tolist() == [index % 2 == 0 for index in range(21)]


def test_forwards_udp_size_policy(tmp_path: Path) -> None:
    source = tmp_path / "vpn_voip_capture1.pcap"
    packet = _packet(protocol=17)
    _write_pcap(source, [(float(index), packet) for index in range(21)])

    release = next(extract_vnat_pcap_windows(source))
    corrected = next(
        extract_vnat_pcap_windows(
            source,
            size_policy=PacketSizePolicy.IP_PACKET,
        )
    )

    assert release.sizes.tolist() == [len(packet) - 20] * 21
    assert corrected.sizes.tolist() == [len(packet)] * 21


def test_forwards_window_threshold_policy(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture1.pcap"
    packet = _packet()
    _write_pcap(source, [(float(index), packet) for index in range(20)])

    release = list(extract_vnat_pcap_windows(source))
    literal = list(
        extract_vnat_pcap_windows(
            source,
            config=WindowExtractionConfig(
                threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
            ),
        )
    )

    assert release == []
    assert len(literal) == 1
    assert literal[0].packet_count == 20


def test_empty_capture_fails_through_shared_window_contract(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture1.pcap"
    _write_pcap(source, [])

    with pytest.raises(VnatWindowError, match="cannot extract windows from an empty capture"):
        list(extract_vnat_pcap_windows(source))
