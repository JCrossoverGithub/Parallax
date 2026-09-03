from collections.abc import Sequence
from pathlib import Path
from socket import inet_aton
from typing import Any, Final, cast

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import (
    IP_PROTOCOL_TCP,
    IP_PROTOCOL_UDP,
    PCAP_LINKTYPE_RAW_IP,
    PacketMetadata,
    PacketSizePolicy,
    PcapReadError,
    iter_pcap_packet_metadata,
)

SOURCE: Final = "10.101.1.100"
DESTINATION: Final = "10.103.1.100"


def _ipv4_packet(
    *,
    protocol: int = IP_PROTOCOL_TCP,
    source: str = SOURCE,
    destination: str = DESTINATION,
    source_port: int = 41_898,
    destination_port: int = 22,
    payload: bytes = b"encrypted-payload",
    more_fragments: bool = False,
) -> bytes:
    if protocol == IP_PROTOCOL_TCP:
        transport = dpkt.tcp.TCP(
            sport=source_port,
            dport=destination_port,
            flags=dpkt.tcp.TH_ACK,
            data=payload,
        )
    elif protocol == IP_PROTOCOL_UDP:
        transport = dpkt.udp.UDP(
            sport=source_port,
            dport=destination_port,
            data=payload,
        )
        transport.ulen = len(transport)
    else:
        transport = payload

    packet = dpkt.ip.IP(
        src=inet_aton(source),
        dst=inet_aton(destination),
        p=protocol,
        ttl=64,
        data=transport,
    )
    packet.mf = int(more_fragments)
    packet.len = len(packet)
    return bytes(packet)


def _write_pcap(
    path: Path,
    records: Sequence[tuple[float, bytes]],
    *,
    linktype: int = PCAP_LINKTYPE_RAW_IP,
) -> None:
    with path.open("wb") as stream:
        writer = dpkt.pcap.Writer(stream, linktype=linktype)
        for timestamp, packet in records:
            writer.writepkt(packet, ts=timestamp)
        writer.close()


def test_reads_raw_ipv4_tcp_and_udp_metadata_without_payload(tmp_path: Path) -> None:
    source = tmp_path / "capture.pcap"
    tcp = _ipv4_packet()
    udp = _ipv4_packet(
        protocol=IP_PROTOCOL_UDP,
        source=DESTINATION,
        destination=SOURCE,
        source_port=22,
        destination_port=41_898,
    )
    _write_pcap(source, [(1.25, tcp), (2.5, udp)])

    packets = list(iter_pcap_packet_metadata(source))

    assert packets == [
        PacketMetadata(
            timestamp_seconds=1.25,
            source_address=SOURCE,
            source_port=41_898,
            destination_address=DESTINATION,
            destination_port=22,
            protocol=IP_PROTOCOL_TCP,
            size=len(tcp),
        ),
        PacketMetadata(
            timestamp_seconds=2.5,
            source_address=DESTINATION,
            source_port=22,
            destination_address=SOURCE,
            destination_port=41_898,
            protocol=IP_PROTOCOL_UDP,
            size=len(udp) - 20,
        ),
    ]
    assert packets[0].connection == (SOURCE, 41_898, DESTINATION, 22, IP_PROTOCOL_TCP)
    assert "encrypted-payload" not in repr(packets)


def test_can_retain_full_ipv4_length_for_udp(tmp_path: Path) -> None:
    source = tmp_path / "corrected-size.pcap"
    udp = _ipv4_packet(protocol=IP_PROTOCOL_UDP)
    _write_pcap(source, [(1.0, udp)])

    packet = next(
        iter_pcap_packet_metadata(
            source,
            size_policy=PacketSizePolicy.IP_PACKET,
        )
    )

    assert packet.size == len(udp)


def test_rejects_invalid_size_policy(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="size_policy must be a PacketSizePolicy"):
        list(
            iter_pcap_packet_metadata(
                tmp_path / "capture.pcap",
                size_policy=cast(Any, "release-compatible"),
            )
        )


def test_rejects_missing_capture(tmp_path: Path) -> None:
    with pytest.raises(PcapReadError, match="PCAP file does not exist"):
        list(iter_pcap_packet_metadata(tmp_path / "missing.pcap"))


def test_rejects_invalid_classic_pcap_header(tmp_path: Path) -> None:
    source = tmp_path / "invalid.pcap"
    source.write_bytes(b"not a pcap")

    with pytest.raises(PcapReadError, match="invalid classic PCAP header"):
        list(iter_pcap_packet_metadata(source))


def test_rejects_unsupported_link_type(tmp_path: Path) -> None:
    source = tmp_path / "ethernet.pcap"
    _write_pcap(source, [(1.0, b"frame")], linktype=dpkt.pcap.DLT_EN10MB)

    with pytest.raises(PcapReadError, match="unsupported PCAP link type 1"):
        list(iter_pcap_packet_metadata(source))


def test_rejects_decreasing_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "unordered.pcap"
    packet = _ipv4_packet()
    _write_pcap(source, [(2.0, packet), (1.0, packet)])

    with pytest.raises(PcapReadError, match="packet 2: timestamp precedes"):
        list(iter_pcap_packet_metadata(source))


@pytest.mark.parametrize(
    ("raw_packet", "message"),
    [
        (b"", "empty Raw-IP record"),
        (b"\x60" + b"\x00" * 39, "unsupported IP version"),
        (b"\x45", "malformed IPv4 packet"),
        (_ipv4_packet(protocol=1), "unsupported IP transport protocol 1"),
        (
            _ipv4_packet(more_fragments=True),
            "fragmented IPv4 packet is unsupported",
        ),
    ],
)
def test_rejects_unsupported_or_malformed_network_packets(
    tmp_path: Path,
    raw_packet: bytes,
    message: str,
) -> None:
    source = tmp_path / "invalid-packet.pcap"
    _write_pcap(source, [(1.0, raw_packet)])

    with pytest.raises(PcapReadError, match=message):
        list(iter_pcap_packet_metadata(source))


def test_rejects_truncated_ipv4_packet(tmp_path: Path) -> None:
    source = tmp_path / "truncated-ip.pcap"
    packet = _ipv4_packet()
    _write_pcap(source, [(1.0, packet[:-1])])

    with pytest.raises(PcapReadError, match="truncated IPv4 packet"):
        list(iter_pcap_packet_metadata(source))


def test_rejects_malformed_transport_header(tmp_path: Path) -> None:
    source = tmp_path / "bad-transport.pcap"
    packet = dpkt.ip.IP(
        src=inet_aton(SOURCE),
        dst=inet_aton(DESTINATION),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=b"short",
    )
    packet.len = len(packet)
    _write_pcap(source, [(1.0, bytes(packet))])

    with pytest.raises(PcapReadError, match="malformed transport header"):
        list(iter_pcap_packet_metadata(source))


def test_rejects_truncated_pcap_record(tmp_path: Path) -> None:
    source = tmp_path / "truncated-record.pcap"
    packet = _ipv4_packet()
    _write_pcap(source, [(1.0, packet)])
    source.write_bytes(source.read_bytes()[:-1])

    with pytest.raises(PcapReadError, match="truncated IPv4 packet"):
        list(iter_pcap_packet_metadata(source))


def test_rejects_truncated_pcap_record_header(tmp_path: Path) -> None:
    source = tmp_path / "truncated-record-header.pcap"
    _write_pcap(source, [])
    source.write_bytes(source.read_bytes() + b"\x00")

    with pytest.raises(PcapReadError, match="could not read PCAP record after packet 0"):
        list(iter_pcap_packet_metadata(source))
