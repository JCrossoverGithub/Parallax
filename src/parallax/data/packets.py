"""Payload-free decoding contracts for supported Raw-IPv4 packets."""

from dataclasses import dataclass
from enum import StrEnum
from socket import AF_INET, inet_ntop
from typing import Final

import dpkt  # type: ignore[import-untyped]

from parallax.data.windowing import ConnectionKey

IP_PROTOCOL_ICMP: Final = 1
IP_PROTOCOL_TCP: Final = 6
IP_PROTOCOL_UDP: Final = 17

_SUPPORTED_IP_PROTOCOLS: Final = (
    IP_PROTOCOL_ICMP,
    IP_PROTOCOL_TCP,
    IP_PROTOCOL_UDP,
)


class PacketDecodeError(ValueError):
    """Raised when Raw-IPv4 bytes cannot become supported packet metadata."""


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


def decode_raw_ipv4_packet(
    timestamp_seconds: float,
    raw_packet: bytes,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
) -> PacketMetadata:
    """Decode supported Raw-IPv4 bytes into payload-free packet metadata."""
    if not isinstance(size_policy, PacketSizePolicy):
        raise TypeError("size_policy must be a PacketSizePolicy")

    if not raw_packet:
        raise PacketDecodeError("empty Raw-IP record")

    if raw_packet[0] >> 4 != 4:
        raise PacketDecodeError("unsupported IP version")

    try:
        network_packet = dpkt.ip.IP(raw_packet)
    except (ValueError, dpkt.dpkt.Error) as error:
        raise PacketDecodeError("malformed IPv4 packet") from error

    packet_size = int(network_packet.len)
    if packet_size > len(raw_packet):
        raise PacketDecodeError("truncated IPv4 packet")

    if network_packet.mf or network_packet.offset:
        raise PacketDecodeError("fragmented IPv4 packet is unsupported")

    protocol = int(network_packet.p)
    if protocol not in _SUPPORTED_IP_PROTOCOLS:
        raise PacketDecodeError(f"unsupported IP protocol {protocol}")

    transport_packet = network_packet.data
    expected_type = {
        IP_PROTOCOL_ICMP: dpkt.icmp.ICMP,
        IP_PROTOCOL_TCP: dpkt.tcp.TCP,
        IP_PROTOCOL_UDP: dpkt.udp.UDP,
    }[protocol]

    if not isinstance(transport_packet, expected_type):
        raise PacketDecodeError("malformed transport header")

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
        timestamp_seconds=timestamp_seconds,
        source_address=inet_ntop(AF_INET, network_packet.src),
        source_port=source_port,
        destination_address=inet_ntop(AF_INET, network_packet.dst),
        destination_port=destination_port,
        protocol=protocol,
        size=selected_size,
    )
