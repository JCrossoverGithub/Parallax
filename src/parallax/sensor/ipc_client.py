"""Unprivileged packet source backed by the local Parallax sensor IPC service."""

import socket
from collections.abc import Callable
from contextlib import suppress
from math import isfinite
from pathlib import Path
from typing import Final, Protocol

from parallax.data import PacketMetadata
from parallax.sensor.ipc_protocol import (
    SENSOR_IPC_MAX_MESSAGE_BYTES,
    SensorErrorMessage,
    SensorIpcMessage,
    SensorIpcProtocolError,
    SensorPacketMessage,
    SensorReadyMessage,
    SensorStartRequest,
    decode_sensor_ipc_message,
    encode_sensor_ipc_message,
)

DEFAULT_SENSOR_IPC_SOCKET_PATH: Final = Path("/run/parallax/sensor.sock")

_DEFAULT_HANDSHAKE_TIMEOUT_SECONDS: Final = 2.0
_DEFAULT_POLL_TIMEOUT_SECONDS: Final = 0.25


class SensorIpcClientError(RuntimeError):
    """Raised when the unprivileged sensor IPC client cannot operate."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code


class SensorIpcClientSocket(Protocol):
    """Socket operations required by the metadata IPC client."""

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        """Set the socket timeout."""

    def connect(
        self,
        address: str,
    ) -> None:
        """Connect to one Unix-domain socket."""

    def sendall(
        self,
        data: bytes,
    ) -> None:
        """Send all bytes."""

    def recv(
        self,
        bufsize: int,
    ) -> bytes:
        """Receive bytes."""

    def close(self) -> None:
        """Close the socket."""


SensorIpcClientSocketFactory = Callable[
    [],
    SensorIpcClientSocket,
]


def _default_socket_factory() -> SensorIpcClientSocket:
    return socket.socket(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )


class SensorIpcPacketSource:
    """Expose privileged sensor metadata through RuntimePacketSource semantics."""

    __slots__ = (
        "_buffer",
        "_connection",
        "_handshake_timeout_seconds",
        "_interface",
        "_poll_timeout_seconds",
        "_socket_factory",
        "_socket_path",
    )

    def __init__(
        self,
        socket_path: str | Path,
        interface: str,
        *,
        handshake_timeout_seconds: float = (_DEFAULT_HANDSHAKE_TIMEOUT_SECONDS),
        poll_timeout_seconds: float = (_DEFAULT_POLL_TIMEOUT_SECONDS),
        socket_factory: SensorIpcClientSocketFactory = (_default_socket_factory),
    ) -> None:
        if not interface.strip():
            raise SensorIpcClientError(
                "sensor interface must not be empty",
                code="sensor_configuration_error",
            )

        _validate_timeout(
            handshake_timeout_seconds,
            description="sensor IPC handshake timeout",
        )
        _validate_timeout(
            poll_timeout_seconds,
            description="sensor IPC poll timeout",
        )

        self._socket_path = Path(socket_path)
        self._interface = interface
        self._handshake_timeout_seconds = handshake_timeout_seconds
        self._poll_timeout_seconds = poll_timeout_seconds
        self._socket_factory = socket_factory
        self._connection: SensorIpcClientSocket | None = None
        self._buffer = bytearray()

    @property
    def socket_path(self) -> Path:
        """Return the configured sensor IPC path."""
        return self._socket_path

    @property
    def interface(self) -> str:
        """Return the requested local capture interface."""
        return self._interface

    @property
    def is_open(self) -> bool:
        """Return whether the IPC session is active."""
        return self._connection is not None

    def open(self) -> None:
        """Connect and complete the sensor start/ready handshake."""
        if self._connection is not None:
            raise SensorIpcClientError(
                "sensor IPC packet source is already open",
                code="sensor_state_error",
            )

        self._buffer.clear()
        connection = self._socket_factory()

        try:
            connection.settimeout(self._handshake_timeout_seconds)
            connection.connect(str(self._socket_path))
            connection.sendall(
                encode_sensor_ipc_message(SensorStartRequest(interface=self._interface))
            )

            message = self._receive_message(
                connection,
                allow_timeout=False,
            )

            if isinstance(
                message,
                SensorErrorMessage,
            ):
                raise SensorIpcClientError(
                    message.message,
                    code=message.code,
                )

            if not isinstance(
                message,
                SensorReadyMessage,
            ):
                raise SensorIpcClientError(
                    "sensor IPC server did not send a ready message",
                    code="sensor_protocol_error",
                )

            if message.interface != self._interface:
                raise SensorIpcClientError(
                    "sensor IPC ready interface does not match request",
                    code="sensor_protocol_error",
                )

            connection.settimeout(self._poll_timeout_seconds)
        except SensorIpcClientError:
            _close_ignoring_error(connection)
            self._buffer.clear()
            raise
        except SensorIpcProtocolError as error:
            _close_ignoring_error(connection)
            self._buffer.clear()
            raise SensorIpcClientError(
                "sensor IPC server sent an invalid message",
                code="sensor_protocol_error",
            ) from error
        except OSError as error:
            _close_ignoring_error(connection)
            self._buffer.clear()
            raise SensorIpcClientError(
                "could not open sensor IPC packet source",
                code="sensor_unavailable",
            ) from error

        self._connection = connection

    def receive(self) -> PacketMetadata | None:
        """Return one packet, or None when the IPC poll is quiet."""
        connection = self._connection

        if connection is None:
            raise SensorIpcClientError(
                "sensor IPC packet source is not open",
                code="sensor_state_error",
            )

        try:
            message = self._receive_message(
                connection,
                allow_timeout=True,
            )
        except SensorIpcProtocolError as error:
            raise SensorIpcClientError(
                "sensor IPC server sent an invalid message",
                code="sensor_protocol_error",
            ) from error

        if message is None:
            return None

        if isinstance(
            message,
            SensorPacketMessage,
        ):
            return message.packet

        if isinstance(
            message,
            SensorErrorMessage,
        ):
            raise SensorIpcClientError(
                message.message,
                code=message.code,
            )

        raise SensorIpcClientError(
            "sensor IPC server sent an unexpected message",
            code="sensor_protocol_error",
        )

    def close(self) -> None:
        """Close the client; EOF tells the privileged sensor to stop."""
        connection = self._connection
        self._connection = None
        self._buffer.clear()

        if connection is None:
            return

        try:
            connection.close()
        except OSError as error:
            raise SensorIpcClientError(
                "could not close sensor IPC packet source",
                code="sensor_ipc_error",
            ) from error

    def _receive_message(
        self,
        connection: SensorIpcClientSocket,
        *,
        allow_timeout: bool,
    ) -> SensorIpcMessage | None:
        while True:
            framed = self._pop_buffered_frame()

            if framed is not None:
                return decode_sensor_ipc_message(framed)

            remaining = SENSOR_IPC_MAX_MESSAGE_BYTES - len(self._buffer)

            try:
                chunk = connection.recv(
                    min(
                        remaining,
                        SENSOR_IPC_MAX_MESSAGE_BYTES,
                    )
                )
            except TimeoutError as error:
                if allow_timeout:
                    return None

                raise SensorIpcClientError(
                    "timed out waiting for sensor IPC response",
                    code="sensor_timeout",
                ) from error
            except OSError as error:
                raise SensorIpcClientError(
                    "could not receive sensor IPC message",
                    code="sensor_ipc_error",
                ) from error

            if not chunk:
                raise SensorIpcClientError(
                    "sensor IPC server disconnected",
                    code="sensor_disconnected",
                )

            self._buffer.extend(chunk)

    def _pop_buffered_frame(
        self,
    ) -> bytes | None:
        newline_index = self._buffer.find(b"\n")

        if newline_index < 0:
            if len(self._buffer) >= SENSOR_IPC_MAX_MESSAGE_BYTES:
                raise SensorIpcProtocolError("sensor IPC message exceeds the maximum size")

            return None

        frame_end = newline_index + 1

        framed = bytes(self._buffer[:frame_end])
        del self._buffer[:frame_end]

        return framed


def _validate_timeout(
    value: float,
    *,
    description: str,
) -> None:
    if not isfinite(value) or value <= 0.0:
        raise SensorIpcClientError(
            f"{description} must be finite and positive",
            code="sensor_configuration_error",
        )


def _close_ignoring_error(
    connection: SensorIpcClientSocket,
) -> None:
    with suppress(OSError):
        connection.close()
