from collections.abc import Sequence
from pathlib import Path
from socket import inet_aton

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data.pcap import (
    IP_PROTOCOL_TCP,
    IP_PROTOCOL_UDP,
    PCAP_LINKTYPE_RAW_IP,
    PacketSizePolicy,
    PcapReadError,
)
from parallax.replay.domain import ReplayConfiguration
from parallax.replay.pcap import iter_pcap_replay_entries

SOURCE = "10.101.1.100"
DESTINATION = "10.103.1.100"


def _ipv4_packet(
    *,
    protocol: int = IP_PROTOCOL_TCP,
    source_port: int = 41_898,
    destination_port: int = 22,
) -> bytes:
    if protocol == IP_PROTOCOL_TCP:
        transport = dpkt.tcp.TCP(
            sport=source_port,
            dport=destination_port,
            flags=dpkt.tcp.TH_ACK,
            data=b"encrypted",
        )
    else:
        transport = dpkt.udp.UDP(
            sport=source_port,
            dport=destination_port,
            data=b"encrypted",
        )
        transport.ulen = len(transport)

    packet = dpkt.ip.IP(
        src=inet_aton(SOURCE),
        dst=inet_aton(DESTINATION),
        p=protocol,
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
        writer = dpkt.pcap.Writer(stream, linktype=PCAP_LINKTYPE_RAW_IP)
        for timestamp, packet in records:
            writer.writepkt(packet, ts=timestamp)
        writer.close()


def test_pcap_replay_entries_attach_packets_to_scaled_schedule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "capture.pcap"
    first_packet = _ipv4_packet()
    second_packet = _ipv4_packet(source_port=41_899)
    _write_pcap(
        source,
        [
            (100.0, first_packet),
            (102.0, second_packet),
        ],
    )

    entries = list(
        iter_pcap_replay_entries(
            source,
            configuration=ReplayConfiguration(time_scale=2.0),
        )
    )

    assert [entry.schedule.packet_number for entry in entries] == [1, 2]
    assert [entry.schedule.source_timestamp_seconds for entry in entries] == [
        100.0,
        102.0,
    ]
    assert [entry.schedule.source_offset_seconds for entry in entries] == [0.0, 2.0]
    assert [entry.schedule.scheduled_offset_seconds for entry in entries] == [0.0, 1.0]
    assert [entry.packet.timestamp_seconds for entry in entries] == [100.0, 102.0]
    assert [entry.packet.source_port for entry in entries] == [41_898, 41_899]

    for entry in entries:
        assert entry.schedule.source_timestamp_seconds == entry.packet.timestamp_seconds


def test_pcap_replay_entries_forward_packet_size_policy(tmp_path: Path) -> None:
    source = tmp_path / "udp.pcap"
    udp_packet = _ipv4_packet(protocol=IP_PROTOCOL_UDP)
    _write_pcap(source, [(1.0, udp_packet)])

    default_entry = next(iter_pcap_replay_entries(source))
    corrected_entry = next(
        iter_pcap_replay_entries(
            source,
            size_policy=PacketSizePolicy.IP_PACKET,
        )
    )

    assert default_entry.packet.size == len(udp_packet) - 20
    assert corrected_entry.packet.size == len(udp_packet)


def test_pcap_replay_entries_remain_lazy_across_parser_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "lazy.pcap"
    _write_pcap(
        source,
        [
            (1.0, _ipv4_packet()),
            (2.0, b"\x60" + b"\x00" * 39),
        ],
    )

    entries = iter_pcap_replay_entries(source)

    first = next(entries)

    assert first.schedule.packet_number == 1
    assert first.packet.timestamp_seconds == 1.0

    with pytest.raises(PcapReadError, match="packet 2: unsupported IP version"):
        next(entries)


def test_empty_pcap_produces_no_replay_entries(tmp_path: Path) -> None:
    source = tmp_path / "empty.pcap"
    _write_pcap(source, [])

    assert list(iter_pcap_replay_entries(source)) == []


def test_pcap_replay_entries_propagate_missing_capture_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(PcapReadError, match="PCAP file does not exist"):
        list(iter_pcap_replay_entries(tmp_path / "missing.pcap"))
