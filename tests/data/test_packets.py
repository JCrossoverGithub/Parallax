from socket import inet_aton
from typing import Any, cast

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import (
    IP_PROTOCOL_TCP,
    IP_PROTOCOL_UDP,
    PacketMetadata,
    PacketSizePolicy,
    decode_raw_ipv4_packet,
)


def _ipv4_packet(
    *,
    protocol: int,
    source_port: int = 41_898,
    destination_port: int = 443,
    payload: bytes = b"opaque-payload",
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
        raise ValueError(f"unsupported test protocol: {protocol}")

    packet = dpkt.ip.IP(
        src=inet_aton("10.0.0.10"),
        dst=inet_aton("10.0.0.20"),
        p=protocol,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def test_decodes_raw_ipv4_bytes_without_retaining_payload() -> None:
    raw_packet = _ipv4_packet(protocol=IP_PROTOCOL_TCP)

    packet = decode_raw_ipv4_packet(12.5, raw_packet)

    assert packet == PacketMetadata(
        timestamp_seconds=12.5,
        source_address="10.0.0.10",
        source_port=41_898,
        destination_address="10.0.0.20",
        destination_port=443,
        protocol=IP_PROTOCOL_TCP,
        size=len(raw_packet),
    )
    assert "opaque-payload" not in repr(packet)


def test_release_compatible_udp_size_matches_vnat_contract() -> None:
    raw_packet = _ipv4_packet(protocol=IP_PROTOCOL_UDP)

    packet = decode_raw_ipv4_packet(1.0, raw_packet)

    assert packet.size == len(raw_packet) - 20


def test_ip_packet_policy_retains_full_udp_ipv4_size() -> None:
    raw_packet = _ipv4_packet(protocol=IP_PROTOCOL_UDP)

    packet = decode_raw_ipv4_packet(
        1.0,
        raw_packet,
        size_policy=PacketSizePolicy.IP_PACKET,
    )

    assert packet.size == len(raw_packet)


def test_decoder_rejects_invalid_size_policy() -> None:
    raw_packet = _ipv4_packet(protocol=IP_PROTOCOL_TCP)

    with pytest.raises(
        TypeError,
        match="size_policy must be a PacketSizePolicy",
    ):
        decode_raw_ipv4_packet(
            1.0,
            raw_packet,
            size_policy=cast(Any, "release-compatible"),
        )
