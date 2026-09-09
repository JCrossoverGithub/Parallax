from collections import deque
from socket import inet_aton
from typing import Any, cast

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import (
    IP_PROTOCOL_UDP,
    PacketSizePolicy,
)
from parallax.sensor import (
    CaptureInterface,
    LiveEthernetCapture,
    LivePacketSource,
    LivePacketSourceStats,
)

_DESTINATION_MAC = bytes.fromhex("001122334455")
_SOURCE_MAC = bytes.fromhex("66778899aabb")


class QueueSocket:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = deque(frames)
        self.closed = False
        self.bound_address: tuple[str, int] | None = None

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_address = address

    def recv(self, bufsize: int) -> bytes:
        assert bufsize == 65_535
        if not self.frames:
            raise AssertionError("test capture frame queue exhausted")
        return self.frames.popleft()

    def close(self) -> None:
        self.closed = True


def _interface() -> CaptureInterface:
    return CaptureInterface(index=2, name="eth0")


def _ipv4_udp_packet() -> bytes:
    transport = dpkt.udp.UDP(
        sport=50_000,
        dport=443,
        data=b"opaque-payload",
    )
    transport.ulen = len(transport)

    packet = dpkt.ip.IP(
        src=inet_aton("10.0.0.10"),
        dst=inet_aton("10.0.0.20"),
        p=IP_PROTOCOL_UDP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _ethernet_frame(
    payload: bytes,
    *,
    ether_type: bytes = b"\x08\x00",
) -> bytes:
    return _DESTINATION_MAC + _SOURCE_MAC + ether_type + payload


def _source(
    frames: list[bytes],
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
) -> tuple[LivePacketSource, QueueSocket]:
    fake_socket = QueueSocket(frames)

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
        clock=lambda: 123.5,
    )

    return (
        LivePacketSource(
            capture,
            size_policy=size_policy,
        ),
        fake_socket,
    )


def test_rejects_invalid_size_policy() -> None:
    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: QueueSocket([]),
    )

    with pytest.raises(
        TypeError,
        match="size_policy must be a PacketSizePolicy",
    ):
        LivePacketSource(
            capture,
            size_policy=cast(Any, "release-compatible"),
        )


def test_exposes_interface_and_open_state() -> None:
    source, fake_socket = _source([])

    assert source.interface == _interface()
    assert not source.is_open

    source.open()

    assert source.is_open
    assert fake_socket.bound_address == ("eth0", 0)

    source.close()

    assert not source.is_open


def test_receives_supported_packet_and_updates_stats() -> None:
    raw_ip = _ipv4_udp_packet()
    source, _ = _source([_ethernet_frame(raw_ip)])

    source.open()
    packet = source.receive()

    assert packet.timestamp_seconds == 123.5
    assert packet.source_address == "10.0.0.10"
    assert packet.source_port == 50_000
    assert packet.destination_address == "10.0.0.20"
    assert packet.destination_port == 443
    assert packet.protocol == IP_PROTOCOL_UDP
    assert packet.size == len(raw_ip) - 20

    assert source.stats == LivePacketSourceStats(
        frames_received=1,
        packets_decoded=1,
        non_ipv4_frames=0,
        invalid_frames=0,
    )

    source.close()


def test_skips_irrelevant_and_invalid_frames() -> None:
    arp_frame = _ethernet_frame(
        b"arp-payload",
        ether_type=b"\x08\x06",
    )
    truncated_frame = b"\x00" * 13
    ipv4_frame = _ethernet_frame(_ipv4_udp_packet())

    source, _ = _source(
        [
            arp_frame,
            truncated_frame,
            ipv4_frame,
        ]
    )

    source.open()
    packet = source.receive()

    assert packet.protocol == IP_PROTOCOL_UDP
    assert source.stats == LivePacketSourceStats(
        frames_received=3,
        packets_decoded=1,
        non_ipv4_frames=1,
        invalid_frames=1,
    )

    source.close()


def test_ip_packet_size_policy_is_forwarded_to_decoder() -> None:
    raw_ip = _ipv4_udp_packet()

    source, _ = _source(
        [_ethernet_frame(raw_ip)],
        size_policy=PacketSizePolicy.IP_PACKET,
    )

    source.open()
    packet = source.receive()

    assert packet.size == len(raw_ip)

    source.close()


def test_open_resets_session_statistics() -> None:
    frame = _ethernet_frame(_ipv4_udp_packet())
    source, _ = _source([frame, frame])

    source.open()
    source.receive()
    source.close()

    assert source.stats.packets_decoded == 1

    source.open()

    assert source.stats == LivePacketSourceStats(
        frames_received=0,
        packets_decoded=0,
        non_ipv4_frames=0,
        invalid_frames=0,
    )

    source.receive()
    source.close()


def test_context_manager_owns_capture_lifecycle() -> None:
    source, fake_socket = _source([_ethernet_frame(_ipv4_udp_packet())])

    with source as active_source:
        assert active_source is source
        assert source.is_open
        assert active_source.receive().protocol == IP_PROTOCOL_UDP

    assert fake_socket.closed
    assert not source.is_open
