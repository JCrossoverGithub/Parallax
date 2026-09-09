"""Ethernet-frame adaptation for the live Parallax sensor."""

from typing import Final

from parallax.data.packets import (
    PacketDecodeError,
    PacketMetadata,
    PacketSizePolicy,
    decode_raw_ipv4_packet,
)

_ETHERNET_HEADER_BYTES: Final = 14

_ETHERTYPE_IPV4: Final = 0x0800
_ETHERTYPE_VLAN: Final = 0x8100
_ETHERTYPE_PROVIDER_VLAN: Final = 0x88A8

_VLAN_TAG_BYTES: Final = 4
_VLAN_ETHERTYPES: Final = {
    _ETHERTYPE_VLAN,
    _ETHERTYPE_PROVIDER_VLAN,
}


class EthernetFrameError(ValueError):
    """Raised when an Ethernet frame cannot be safely interpreted."""


def decode_ethernet_ipv4_frame(
    timestamp_seconds: float,
    raw_frame: bytes,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
) -> PacketMetadata | None:
    """Convert one Ethernet IPv4 frame to shared packet metadata.

    Non-IPv4 Ethernet traffic is intentionally ignored by returning ``None``.
    """
    if len(raw_frame) < _ETHERNET_HEADER_BYTES:
        raise EthernetFrameError("truncated Ethernet frame")

    ether_type = int.from_bytes(raw_frame[12:14], byteorder="big")
    payload_offset = _ETHERNET_HEADER_BYTES

    while ether_type in _VLAN_ETHERTYPES:
        if len(raw_frame) < payload_offset + _VLAN_TAG_BYTES:
            raise EthernetFrameError("truncated Ethernet VLAN tag")

        ether_type = int.from_bytes(
            raw_frame[payload_offset + 2 : payload_offset + 4],
            byteorder="big",
        )
        payload_offset += _VLAN_TAG_BYTES

    if ether_type != _ETHERTYPE_IPV4:
        return None

    try:
        return decode_raw_ipv4_packet(
            timestamp_seconds,
            raw_frame[payload_offset:],
            size_policy=size_policy,
        )
    except PacketDecodeError as error:
        raise EthernetFrameError(f"invalid IPv4 Ethernet payload: {error}") from error
