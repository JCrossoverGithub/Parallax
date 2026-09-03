"""Strict metadata-only parsing for classic Raw-IP PCAP captures."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from socket import AF_INET, inet_ntop
from typing import Final

import dpkt  # type: ignore[import-untyped]

from parallax.data.windowing import ConnectionKey

PCAP_LINKTYPE_RAW_IP: Final = 101
IP_PROTOCOL_ICMP: Final = 1
IP_PROTOCOL_TCP: Final = 6
IP_PROTOCOL_UDP: Final = 17
_SUPPORTED_IP_PROTOCOLS: Final = (
    IP_PROTOCOL_ICMP,
    IP_PROTOCOL_TCP,
    IP_PROTOCOL_UDP,
)


class PcapReadError(ValueError):
    """Raised when a PCAP cannot be converted to supported packet metadata."""


class PacketSizePolicy(StrEnum):
    """Packet-size interpretation used by downstream window features."""

    RELEASE_COMPATIBLE = "release-compatible"
    IP_PACKET = "ip-packet"


@dataclass(frozen=True, slots=True)
class PacketMetadata:
    """Payload-free metadata retained from one supported IP packet."""

    timestamp_seconds: float
    source_address: str
    source_port: int
    destination_address: str
    destination_port: int
    protocol: int
    size: int

    @property
    def connection(self) -> ConnectionKey:
        """Return the packet's directional five-tuple."""
        return (
            self.source_address,
            self.source_port,
            self.destination_address,
            self.destination_port,
            self.protocol,
        )


def iter_pcap_packet_metadata(
    source: str | Path,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
) -> Iterator[PacketMetadata]:
    """Yield supported packet metadata from one timestamp-ordered classic PCAP."""
    if not isinstance(size_policy, PacketSizePolicy):
        raise TypeError("size_policy must be a PacketSizePolicy")

    source_path = Path(source)
    if not source_path.is_file():
        raise PcapReadError(f"PCAP file does not exist: {source_path}")

    with source_path.open("rb") as stream:
        try:
            reader = dpkt.pcap.Reader(stream)
        except (ValueError, dpkt.dpkt.Error) as error:
            raise PcapReadError(f"invalid classic PCAP header: {source_path}") from error

        linktype = int(reader.datalink())
        if linktype != PCAP_LINKTYPE_RAW_IP:
            raise PcapReadError(
                f"unsupported PCAP link type {linktype}; expected Raw IP ({PCAP_LINKTYPE_RAW_IP})"
            )

        previous_timestamp: float | None = None
        packets_read = 0
        try:
            for packet_number, (raw_timestamp, raw_packet) in enumerate(reader, start=1):
                timestamp = float(raw_timestamp)
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise PcapReadError(
                        f"packet {packet_number}: timestamp precedes the previous packet"
                    )
                previous_timestamp = timestamp
                yield _parse_raw_ipv4_packet(
                    timestamp,
                    bytes(raw_packet),
                    packet_number,
                    size_policy,
                )
                packets_read = packet_number
        except dpkt.dpkt.Error as error:
            raise PcapReadError(
                f"could not read PCAP record after packet {packets_read}"
            ) from error


def _parse_raw_ipv4_packet(
    timestamp: float,
    raw_packet: bytes,
    packet_number: int,
    size_policy: PacketSizePolicy,
) -> PacketMetadata:
    if not raw_packet:
        raise PcapReadError(f"packet {packet_number}: empty Raw-IP record")
    if raw_packet[0] >> 4 != 4:
        raise PcapReadError(f"packet {packet_number}: unsupported IP version")

    try:
        network_packet = dpkt.ip.IP(raw_packet)
    except (ValueError, dpkt.dpkt.Error) as error:
        raise PcapReadError(f"packet {packet_number}: malformed IPv4 packet") from error

    packet_size = int(network_packet.len)
    if packet_size > len(raw_packet):
        raise PcapReadError(f"packet {packet_number}: truncated IPv4 packet")
    if network_packet.mf or network_packet.offset:
        raise PcapReadError(f"packet {packet_number}: fragmented IPv4 packet is unsupported")

    protocol = int(network_packet.p)
    if protocol not in _SUPPORTED_IP_PROTOCOLS:
        raise PcapReadError(f"packet {packet_number}: unsupported IP protocol {protocol}")

    transport_packet = network_packet.data
    expected_type = {
        IP_PROTOCOL_ICMP: dpkt.icmp.ICMP,
        IP_PROTOCOL_TCP: dpkt.tcp.TCP,
        IP_PROTOCOL_UDP: dpkt.udp.UDP,
    }[protocol]
    if not isinstance(transport_packet, expected_type):
        raise PcapReadError(f"packet {packet_number}: malformed transport header")

    source_port = 0
    destination_port = 0
    if protocol != IP_PROTOCOL_ICMP:
        source_port = int(transport_packet.sport)
        destination_port = int(transport_packet.dport)

    # VNAT release 1 records UDP datagram length without its IPv4 header while
    # retaining full IPv4 length for TCP. Preserve that asymmetry only when
    # producing inputs for the published model contract.
    selected_size = packet_size
    if size_policy is PacketSizePolicy.RELEASE_COMPATIBLE and protocol == IP_PROTOCOL_UDP:
        selected_size = int(transport_packet.ulen)

    return PacketMetadata(
        timestamp_seconds=timestamp,
        source_address=inet_ntop(AF_INET, network_packet.src),
        source_port=source_port,
        destination_address=inet_ntop(AF_INET, network_packet.dst),
        destination_port=destination_port,
        protocol=protocol,
        size=selected_size,
    )
