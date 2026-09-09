"""Least-privilege Unix-domain socket server for live packet metadata."""

import socket
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from parallax.data import PacketMetadata
from parallax.sensor.capture import (
    LiveEthernetCapture,
    SensorCaptureError,
)
from parallax.sensor.interfaces import (
    CaptureInterface,
    SensorInterfaceError,
    resolve_capture_interface,
)
from parallax.sensor.ipc_protocol import (
    SENSOR_IPC_MAX_MESSAGE_BYTES,
    SensorErrorMessage,
    SensorIpcProtocolError,
    SensorPacketMessage,
    SensorReadyMessage,
    SensorStartRequest,
    decode_sensor_ipc_message,
    encode_sensor_ipc_message,
)
from parallax.sensor.source import LivePacketSource


class SensorIpcServerError(RuntimeError):
    """Raised when the local sensor server cannot operate safely."""


class SensorPacketSource(Protocol):
    """Packet-source contract owned by one privileged sensor session."""

    def open(self) -> None:
        """Open packet capture."""

    def receive(self) -> PacketMetadata | None:
        """Return one packet or None after a quiet capture poll."""

    def close(self) -> None:
        """Release packet-capture resources."""


SensorPacketSourceFactory = Callable[
    [CaptureInterface],
    SensorPacketSource,
]

SensorInterfaceResolver = Callable[
    [str],
    CaptureInterface,
]


def _default_source_factory(
    interface: CaptureInterface,
) -> SensorPacketSource:
    return LivePacketSource(LiveEthernetCapture(interface))


class SensorIpcSession:
    """Serve one metadata-only sensor client connection."""

    __slots__ = (
        "_connection",
        "_interface_resolver",
        "_source_factory",
    )

    def __init__(
        self,
        connection: socket.socket,
        *,
        interface_resolver: SensorInterfaceResolver = (resolve_capture_interface),
        source_factory: SensorPacketSourceFactory = (_default_source_factory),
    ) -> None:
        self._connection = connection
        self._interface_resolver = interface_resolver
        self._source_factory = source_factory

    def run(self) -> None:
        """Serve exactly one start request until the client disconnects."""
        try:
            request = self._receive_start_request()
        except SensorIpcProtocolError as error:
            self._try_send_error(
                "invalid_request",
                str(error),
            )
            return

        try:
            interface = self._interface_resolver(request.interface)
        except SensorInterfaceError as error:
            self._try_send_error(
                "interface_error",
                str(error),
            )
            return

        source = self._source_factory(interface)
        opened = False

        try:
            source.open()
            opened = True

            if not self._try_send(
                SensorReadyMessage(
                    interface=interface.name,
                )
            ):
                return

            try:
                while True:
                    if self._client_disconnected():
                        return

                    packet = source.receive()

                    if packet is None:
                        continue

                    if self._client_disconnected():
                        return

                    if not self._try_send(SensorPacketMessage(packet=packet)):
                        return
            except SensorIpcProtocolError as error:
                self._try_send_error(
                    "protocol_error",
                    str(error),
                )

        except SensorCaptureError as error:
            self._try_send_error(
                "capture_error",
                str(error),
            )
        finally:
            if opened:
                source.close()

    def _receive_start_request(
        self,
    ) -> SensorStartRequest:
        framed = self._receive_frame()
        message = decode_sensor_ipc_message(framed)

        if not isinstance(
            message,
            SensorStartRequest,
        ):
            raise SensorIpcProtocolError("first sensor IPC message must be a start request")

        return message

    def _receive_frame(self) -> bytes:
        buffer = bytearray()

        while True:
            remaining = SENSOR_IPC_MAX_MESSAGE_BYTES - len(buffer)

            if remaining <= 0:
                raise SensorIpcProtocolError("sensor IPC message exceeds the maximum size")

            chunk = self._connection.recv(min(remaining, 1_024))

            if not chunk:
                raise SensorIpcProtocolError(
                    "sensor IPC client disconnected before sending a request"
                )

            buffer.extend(chunk)

            newline_index = buffer.find(b"\n")

            if newline_index < 0:
                continue

            if newline_index != len(buffer) - 1:
                raise SensorIpcProtocolError(
                    "sensor IPC start request must contain exactly one message"
                )

            return bytes(buffer)

    def _client_disconnected(self) -> bool:
        try:
            pending = self._connection.recv(
                1,
                socket.MSG_PEEK | socket.MSG_DONTWAIT,
            )
        except BlockingIOError:
            return False
        except OSError:
            return True

        if pending == b"":
            return True

        raise SensorIpcProtocolError("sensor IPC client sent unexpected data after start")

    def _try_send(
        self,
        message: (SensorReadyMessage | SensorPacketMessage | SensorErrorMessage),
    ) -> bool:
        try:
            self._connection.sendall(encode_sensor_ipc_message(message))
        except OSError:
            return False

        return True

    def _try_send_error(
        self,
        code: str,
        message: str,
    ) -> None:
        self._try_send(
            SensorErrorMessage(
                code=code,
                message=message,
            )
        )


class UnixSensorServer:
    """Own a local AF_UNIX listener for sequential sensor sessions."""

    __slots__ = (
        "_interface_resolver",
        "_listener",
        "_socket_path",
        "_source_factory",
    )

    def __init__(
        self,
        socket_path: str | Path,
        *,
        interface_resolver: SensorInterfaceResolver = (resolve_capture_interface),
        source_factory: SensorPacketSourceFactory = (_default_source_factory),
    ) -> None:
        self._socket_path = Path(socket_path)
        self._interface_resolver = interface_resolver
        self._source_factory = source_factory
        self._listener: socket.socket | None = None

    @property
    def socket_path(self) -> Path:
        """Return the configured local IPC path."""
        return self._socket_path

    @property
    def is_open(self) -> bool:
        """Return whether the listening socket is active."""
        return self._listener is not None

    def open(self) -> None:
        """Bind the Unix-domain socket without replacing an existing path."""
        if self._listener is not None:
            raise SensorIpcServerError("sensor IPC server is already open")

        if self._socket_path.exists():
            raise SensorIpcServerError(
                f"sensor IPC socket path already exists: {self._socket_path}"
            )

        self._socket_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        listener = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )

        try:
            listener.bind(str(self._socket_path))
            listener.listen(1)
        except OSError as error:
            listener.close()

            if self._socket_path.exists():
                self._socket_path.unlink()

            raise SensorIpcServerError("could not open sensor IPC server") from error

        self._listener = listener

    def serve_one(self) -> None:
        """Accept and serve one client connection."""
        listener = self._listener

        if listener is None:
            raise SensorIpcServerError("sensor IPC server is not open")

        try:
            connection, _ = listener.accept()
        except OSError as error:
            raise SensorIpcServerError("could not accept sensor IPC client") from error

        with connection:
            try:
                SensorIpcSession(
                    connection,
                    interface_resolver=(self._interface_resolver),
                    source_factory=self._source_factory,
                ).run()
            except SensorCaptureError:
                return

    def close(self) -> None:
        """Close the listener and remove its Unix-domain socket path."""
        listener = self._listener
        self._listener = None

        close_error: OSError | None = None

        if listener is not None:
            try:
                listener.close()
            except OSError as error:
                close_error = error

        try:
            if self._socket_path.exists():
                self._socket_path.unlink()
        except OSError as error:
            if close_error is None:
                close_error = error

        if close_error is not None:
            raise SensorIpcServerError("could not close sensor IPC server") from close_error

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
