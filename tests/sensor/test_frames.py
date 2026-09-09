from socket import inet_aton

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import (
    IP_PROTOCOL_TCP,
    PacketMetadata,
    PacketSizePolicy,
)
from parallax.sensor import (
    EthernetFrameError,
    decode_ethernet_ipv4_frame,
)

_DESTINATION_MAC = bytes.fromhex("001122334455")
_SOURCE_MAC = bytes.fromhex("66778899aabb")

_ETHERTYPE_IPV4 = b"\x08\x00"
_ETHERTYPE_ARP = b"\x08\x06"
_ETHERTYPE_VLAN = b"\x81\x00"


def _ipv4_tcp_packet() -> bytes:
    transport = dpkt.tcp.TCP(
        sport=41_898,
        dport=443,
        flags=dpkt.tcp.TH_ACK,
        data=b"opaque-payload",
    )

    packet = dpkt.ip.IP(
        src=inet_aton("10.0.0.10"),
        dst=inet_aton("10.0.0.20"),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _ethernet_frame(
    payload: bytes,
    *,
    ether_type: bytes = _ETHERTYPE_IPV4,
) -> bytes:
    return _DESTINATION_MAC + _SOURCE_MAC + ether_type + payload


def test_decodes_ethernet_ipv4_using_shared_packet_contract() -> None:
    raw_ip = _ipv4_tcp_packet()
    frame = _ethernet_frame(raw_ip)

    packet = decode_ethernet_ipv4_frame(12.5, frame)

    assert packet == PacketMetadata(
        timestamp_seconds=12.5,
        source_address="10.0.0.10",
        source_port=41_898,
        destination_address="10.0.0.20",
        destination_port=443,
        protocol=IP_PROTOCOL_TCP,
        size=len(raw_ip),
    )


def test_non_ipv4_ethernet_traffic_is_ignored() -> None:
    frame = _ethernet_frame(
        b"not-an-ip-packet",
        ether_type=_ETHERTYPE_ARP,
    )

    assert decode_ethernet_ipv4_frame(1.0, frame) is None


def test_decodes_single_vlan_tagged_ipv4_frame() -> None:
    raw_ip = _ipv4_tcp_packet()

    frame = (
        _DESTINATION_MAC + _SOURCE_MAC + _ETHERTYPE_VLAN + b"\x00\x2a" + _ETHERTYPE_IPV4 + raw_ip
    )

    packet = decode_ethernet_ipv4_frame(
        2.0,
        frame,
        size_policy=PacketSizePolicy.IP_PACKET,
    )

    assert packet is not None
    assert packet.source_address == "10.0.0.10"
    assert packet.destination_address == "10.0.0.20"
    assert packet.size == len(raw_ip)


def test_rejects_truncated_ethernet_frame() -> None:
    with pytest.raises(
        EthernetFrameError,
        match="truncated Ethernet frame",
    ):
        decode_ethernet_ipv4_frame(1.0, b"\x00" * 13)


def test_rejects_truncated_vlan_tag() -> None:
    frame = _DESTINATION_MAC + _SOURCE_MAC + _ETHERTYPE_VLAN + b"\x00"

    with pytest.raises(
        EthernetFrameError,
        match="truncated Ethernet VLAN tag",
    ):
        decode_ethernet_ipv4_frame(1.0, frame)


def test_wraps_invalid_ipv4_payload() -> None:
    frame = _ethernet_frame(b"\x45")

    with pytest.raises(
        EthernetFrameError,
        match="invalid IPv4 Ethernet payload: malformed IPv4 packet",
    ):
        decode_ethernet_ipv4_frame(1.0, frame)
