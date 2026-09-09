"""Linux AF_PACKET capture lifecycle for the live Parallax sensor."""

import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from types import TracebackType
from typing import Protocol, Self

from parallax.sensor.interfaces import CaptureInterface

_ETH_P_ALL = 0x0003
_MAX_CAPTURE_BYTES = 65_535
_DEFAULT_POLL_TIMEOUT_SECONDS = 0.25


class SensorCaptureError(RuntimeError):
    """Raised when a live capture socket cannot be operated safely."""


class CaptureSocket(Protocol):
    """Minimal socket contract required by the live capture source."""

    def bind(self, address: tuple[str, int]) -> None:
        """Bind the socket to one network interface."""

    def settimeout(self, value: float | None) -> None:
        """Set the maximum blocking receive duration."""

    def recv(self, bufsize: int) -> bytes:
        """Receive one raw Ethernet frame."""

    def close(self) -> None:
        """Release the socket."""


CaptureSocketFactory = Callable[[], CaptureSocket]


@dataclass(frozen=True, slots=True)
class CapturedEthernetFrame:
    """One ephemeral Ethernet frame and its local receive timestamp."""

    timestamp_seconds: float
    data: bytes


def _open_linux_packet_socket() -> CaptureSocket:
    return socket.socket(
        socket.AF_PACKET,
        socket.SOCK_RAW,
        socket.htons(_ETH_P_ALL),
    )


class LiveEthernetCapture:
    """Own one bound Linux AF_PACKET socket for an explicit interface."""

    __slots__ = (
        "_clock",
        "_interface",
        "_poll_timeout_seconds",
        "_socket",
        "_socket_factory",
    )

    def __init__(
        self,
        interface: CaptureInterface,
        *,
        socket_factory: CaptureSocketFactory | None = None,
        clock: Callable[[], float] | None = None,
        poll_timeout_seconds: float = _DEFAULT_POLL_TIMEOUT_SECONDS,
    ) -> None:
        if not isfinite(poll_timeout_seconds) or poll_timeout_seconds <= 0.0:
            raise SensorCaptureError("capture poll timeout must be finite and positive")

        self._interface = interface
        self._socket_factory = (
            socket_factory if socket_factory is not None else _open_linux_packet_socket
        )
        self._clock = clock if clock is not None else time.time
        self._poll_timeout_seconds = poll_timeout_seconds
        self._socket: CaptureSocket | None = None

    @property
    def interface(self) -> CaptureInterface:
        """Return the interface selected for this capture source."""
        return self._interface

    @property
    def is_open(self) -> bool:
        """Return whether this source currently owns an open socket."""
        return self._socket is not None

    def open(self) -> None:
        """Open, bind, and configure the capture socket."""
        if self._socket is not None:
            raise SensorCaptureError("live capture socket is already open")

        try:
            capture_socket = self._socket_factory()
        except PermissionError as error:
            raise SensorCaptureError(
                "permission denied opening Linux packet capture socket"
            ) from error
        except OSError as error:
            raise SensorCaptureError("could not open Linux packet capture socket") from error

        try:
            capture_socket.bind((self._interface.name, 0))
        except OSError as error:
            capture_socket.close()
            raise SensorCaptureError(
                f"could not bind capture socket to interface {self._interface.name}"
            ) from error

        try:
            capture_socket.settimeout(self._poll_timeout_seconds)
        except OSError as error:
            capture_socket.close()
            raise SensorCaptureError("could not configure live capture poll timeout") from error

        self._socket = capture_socket

    def receive(self) -> CapturedEthernetFrame | None:
        """Receive one frame, or return None when the poll interval expires."""
        capture_socket = self._socket

        if capture_socket is None:
            raise SensorCaptureError("live capture socket is not open")

        try:
            data = capture_socket.recv(_MAX_CAPTURE_BYTES)
        except TimeoutError:
            return None
        except OSError as error:
            raise SensorCaptureError("could not receive Ethernet frame") from error

        return CapturedEthernetFrame(
            timestamp_seconds=self._clock(),
            data=data,
        )

    def close(self) -> None:
        """Release the capture socket if one is open."""
        capture_socket = self._socket

        if capture_socket is None:
            return

        self._socket = None

        try:
            capture_socket.close()
        except OSError as error:
            raise SensorCaptureError("could not close live capture socket") from error

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
