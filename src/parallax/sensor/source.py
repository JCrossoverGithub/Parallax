"""PacketMetadata source built on the live Ethernet capture boundary."""

from dataclasses import dataclass
from types import TracebackType
from typing import Self

from parallax.data.packets import PacketMetadata, PacketSizePolicy
from parallax.sensor.capture import LiveEthernetCapture
from parallax.sensor.frames import (
    EthernetFrameError,
    decode_ethernet_ipv4_frame,
)
from parallax.sensor.interfaces import CaptureInterface


@dataclass(frozen=True, slots=True)
class LivePacketSourceStats:
    """Snapshot of one live packet-source session."""

    frames_received: int
    packets_decoded: int
    non_ipv4_frames: int
    invalid_frames: int


class LivePacketSource:
    """Convert live Ethernet frames into supported PacketMetadata."""

    __slots__ = (
        "_capture",
        "_frames_received",
        "_invalid_frames",
        "_non_ipv4_frames",
        "_packets_decoded",
        "_size_policy",
    )

    def __init__(
        self,
        capture: LiveEthernetCapture,
        *,
        size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
    ) -> None:
        if not isinstance(size_policy, PacketSizePolicy):
            raise TypeError("size_policy must be a PacketSizePolicy")

        self._capture = capture
        self._size_policy = size_policy
        self._frames_received = 0
        self._packets_decoded = 0
        self._non_ipv4_frames = 0
        self._invalid_frames = 0

    @property
    def interface(self) -> CaptureInterface:
        """Return the interface owned by the underlying capture."""
        return self._capture.interface

    @property
    def is_open(self) -> bool:
        """Return whether the underlying live capture is open."""
        return self._capture.is_open

    @property
    def stats(self) -> LivePacketSourceStats:
        """Return immutable counters for the current capture session."""
        return LivePacketSourceStats(
            frames_received=self._frames_received,
            packets_decoded=self._packets_decoded,
            non_ipv4_frames=self._non_ipv4_frames,
            invalid_frames=self._invalid_frames,
        )

    def open(self) -> None:
        """Open the underlying capture and begin a fresh statistics session."""
        self._capture.open()

        self._frames_received = 0
        self._packets_decoded = 0
        self._non_ipv4_frames = 0
        self._invalid_frames = 0

    def receive(self) -> PacketMetadata:
        """Return the next supported live IPv4 packet.

        Ethernet traffic that is not IPv4, or IPv4 frames that cannot satisfy
        the current Parallax packet contract, is discarded without retaining
        payload data.
        """
        while True:
            frame = self._capture.receive()
            self._frames_received += 1

            try:
                packet = decode_ethernet_ipv4_frame(
                    frame.timestamp_seconds,
                    frame.data,
                    size_policy=self._size_policy,
                )
            except EthernetFrameError:
                self._invalid_frames += 1
                continue

            if packet is None:
                self._non_ipv4_frames += 1
                continue

            self._packets_decoded += 1
            return packet

    def close(self) -> None:
        """Close the underlying live capture."""
        self._capture.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
