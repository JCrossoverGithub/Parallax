import socket
from collections import deque
from collections.abc import Callable
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import cast

import pytest

from parallax.data import PacketMetadata
from parallax.sensor import (
    CaptureInterface,
    SensorCaptureError,
    SensorInterfaceError,
)
from parallax.sensor.ipc_protocol import (
    SensorErrorMessage,
    SensorIpcMessage,
    SensorPacketMessage,
    SensorReadyMessage,
    SensorStartRequest,
    decode_sensor_ipc_message,
    encode_sensor_ipc_message,
)
from parallax.sensor.ipc_server import (
    SensorIpcServerError,
    SensorIpcSession,
    UnixSensorServer,
)


def _interface() -> CaptureInterface:
    return CaptureInterface(
        index=2,
        name="eth0",
    )


def _packet() -> PacketMetadata:
    return PacketMetadata(
        timestamp_seconds=123.5,
        source_address="10.0.0.10",
        source_port=50_000,
        destination_address="10.0.0.20",
        destination_port=443,
        protocol=6,
        size=1200,
    )


class FakeSource:
    def __init__(
        self,
        results: list[PacketMetadata | None],
        *,
        open_error: SensorCaptureError | None = None,
        receive_error: SensorCaptureError | None = None,
        close_error: SensorCaptureError | None = None,
    ) -> None:
        self.results = deque(results)
        self.open_error = open_error
        self.receive_error = receive_error
        self.close_error = close_error
        self.open_calls = 0
        self.receive_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1

        if self.open_error is not None:
            raise self.open_error

    def receive(self) -> PacketMetadata | None:
        self.receive_calls += 1

        if self.receive_error is not None:
            raise self.receive_error

        if not self.results:
            sleep(0.005)
            return None

        return self.results.popleft()

    def close(self) -> None:
        self.close_calls += 1

        if self.close_error is not None:
            raise self.close_error


class ScriptedConnection:
    """Small socket double for deterministic session lifecycle tests."""

    def __init__(
        self,
        request: bytes,
        *,
        peek_results: list[bytes | OSError] | None = None,
        fail_send_call: int | None = None,
    ) -> None:
        self.request = request
        self.request_consumed = False
        self.peek_results = deque([] if peek_results is None else peek_results)
        self.fail_send_call = fail_send_call
        self.send_calls = 0
        self.sent: list[bytes] = []

    def recv(
        self,
        bufsize: int,
        flags: int = 0,
    ) -> bytes:
        if flags:
            if not self.peek_results:
                raise BlockingIOError("no pending client data")

            result = self.peek_results.popleft()

            if isinstance(result, OSError):
                raise result

            return result

        if self.request_consumed:
            return b""

        self.request_consumed = True
        return self.request[:bufsize]

    def sendall(self, data: bytes) -> None:
        self.send_calls += 1

        if self.fail_send_call == self.send_calls:
            raise BrokenPipeError("client connection closed")

        self.sent.append(data)


class FakeUnixListener:
    """Unix-listener double for deterministic server error tests."""

    def __init__(
        self,
        *,
        create_path_on_bind: bool = False,
        bind_error: OSError | None = None,
        accept_error: OSError | None = None,
        close_error: OSError | None = None,
    ) -> None:
        self.create_path_on_bind = create_path_on_bind
        self.bind_error = bind_error
        self.accept_error = accept_error
        self.close_error = close_error
        self.bound_path: Path | None = None
        self.listen_backlog: int | None = None
        self.closed = False

    def bind(self, address: str) -> None:
        self.bound_path = Path(address)

        if self.create_path_on_bind:
            self.bound_path.touch()

        if self.bind_error is not None:
            raise self.bind_error

    def listen(self, backlog: int) -> None:
        self.listen_backlog = backlog

    def accept(
        self,
    ) -> tuple[socket.socket, object]:
        if self.accept_error is not None:
            raise self.accept_error

        raise AssertionError("fake listener accept result not configured")

    def close(self) -> None:
        self.closed = True

        if self.close_error is not None:
            raise self.close_error


def _recv_message(
    connection: socket.socket,
) -> SensorIpcMessage:
    data = bytearray()

    while not data.endswith(b"\n"):
        chunk = connection.recv(4096)

        if not chunk:
            raise AssertionError("connection closed before IPC message")

        data.extend(chunk)

    return decode_sensor_ipc_message(bytes(data))


def _resolve_interface(
    _name: str,
) -> CaptureInterface:
    return _interface()


def _start_session(
    source: FakeSource,
    *,
    interface_resolver: Callable[
        [str],
        CaptureInterface,
    ] = _resolve_interface,
) -> tuple[
    Thread,
    socket.socket,
    socket.socket,
]:
    server_connection, client_connection = socket.socketpair()

    session = SensorIpcSession(
        server_connection,
        interface_resolver=interface_resolver,
        source_factory=lambda interface: source,
    )

    thread = Thread(
        target=session.run,
        daemon=True,
    )
    thread.start()

    return (
        thread,
        server_connection,
        client_connection,
    )


def _join(thread: Thread) -> None:
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_session_streams_metadata_until_disconnect() -> None:
    source = FakeSource(
        [
            None,
            _packet(),
        ]
    )
    thread, server, client = _start_session(source)

    try:
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client) == (SensorReadyMessage(interface="eth0"))
        assert _recv_message(client) == (SensorPacketMessage(packet=_packet()))

        client.close()
        _join(thread)

        assert source.open_calls == 1
        assert source.receive_calls >= 2
        assert source.close_calls == 1
    finally:
        server.close()


def test_session_rejects_non_start_request() -> None:
    source = FakeSource([])
    thread, server, client = _start_session(source)

    try:
        client.sendall(encode_sensor_ipc_message(SensorReadyMessage(interface="eth0")))

        response = _recv_message(client)

        assert isinstance(
            response,
            SensorErrorMessage,
        )
        assert response.code == "invalid_request"
        assert "start request" in response.message

        _join(thread)
        assert source.open_calls == 0
    finally:
        server.close()
        client.close()


def test_session_rejects_disconnect_before_request() -> None:
    source = FakeSource([])
    thread, server, client = _start_session(source)

    client.close()
    _join(thread)

    assert source.open_calls == 0
    server.close()


def test_session_rejects_oversized_request() -> None:
    source = FakeSource([])
    thread, server, client = _start_session(source)

    try:
        client.sendall(b"x" * 4096)

        response = _recv_message(client)

        assert isinstance(
            response,
            SensorErrorMessage,
        )
        assert response.code == "invalid_request"

        _join(thread)
    finally:
        server.close()
        client.close()


def test_session_rejects_multiple_start_messages() -> None:
    source = FakeSource([])
    thread, server, client = _start_session(source)

    request = encode_sensor_ipc_message(SensorStartRequest(interface="eth0"))

    try:
        client.sendall(request + request)

        response = _recv_message(client)

        assert isinstance(
            response,
            SensorErrorMessage,
        )
        assert response.code == "invalid_request"

        _join(thread)
    finally:
        server.close()
        client.close()


def test_session_reports_interface_error() -> None:
    def fail_interface(
        name: str,
    ) -> CaptureInterface:
        raise SensorInterfaceError(f"capture interface not found: {name}")

    source = FakeSource([])
    thread, server, client = _start_session(
        source,
        interface_resolver=fail_interface,
    )

    try:
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="missing0")))

        assert _recv_message(client) == (
            SensorErrorMessage(
                code="interface_error",
                message=("capture interface not found: missing0"),
            )
        )

        _join(thread)
        assert source.open_calls == 0
    finally:
        server.close()
        client.close()


def test_session_reports_capture_open_error() -> None:
    source = FakeSource(
        [],
        open_error=SensorCaptureError("capture permission denied"),
    )
    thread, server, client = _start_session(source)

    try:
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client) == (
            SensorErrorMessage(
                code="capture_error",
                message="capture permission denied",
            )
        )

        _join(thread)

        assert source.open_calls == 1
        assert source.close_calls == 0
    finally:
        server.close()
        client.close()


def test_session_reports_capture_receive_error() -> None:
    source = FakeSource(
        [],
        receive_error=SensorCaptureError("capture receive failed"),
    )
    thread, server, client = _start_session(source)

    try:
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client) == (SensorReadyMessage(interface="eth0"))
        assert _recv_message(client) == (
            SensorErrorMessage(
                code="capture_error",
                message="capture receive failed",
            )
        )

        _join(thread)
        assert source.close_calls == 1
    finally:
        server.close()
        client.close()


def test_session_rejects_client_data_after_start() -> None:
    source = FakeSource([])
    thread, server, client = _start_session(source)

    try:
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client) == (SensorReadyMessage(interface="eth0"))

        client.sendall(b"unexpected")

        deadline = monotonic() + 1.0

        while thread.is_alive() and monotonic() < deadline:
            sleep(0.01)

        _join(thread)
        assert source.close_calls == 1
    finally:
        server.close()
        client.close()


def test_default_source_factory_composes_live_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource([])
    capture_marker = object()
    captures: list[CaptureInterface] = []
    sources: list[object] = []

    def fake_capture(
        interface: CaptureInterface,
    ) -> object:
        captures.append(interface)
        return capture_marker

    def fake_source(
        capture: object,
    ) -> FakeSource:
        sources.append(capture)
        return source

    monkeypatch.setattr(
        "parallax.sensor.ipc_server.LiveEthernetCapture",
        fake_capture,
    )
    monkeypatch.setattr(
        "parallax.sensor.ipc_server.LivePacketSource",
        fake_source,
    )

    server_connection, client_connection = socket.socketpair()

    session = SensorIpcSession(
        server_connection,
        interface_resolver=_resolve_interface,
    )

    thread = Thread(
        target=session.run,
        daemon=True,
    )
    thread.start()

    try:
        client_connection.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client_connection) == SensorReadyMessage(interface="eth0")

        client_connection.close()
        _join(thread)

        assert captures == [_interface()]
        assert sources == [capture_marker]
        assert source.open_calls == 1
        assert source.close_calls == 1
    finally:
        server_connection.close()


def test_session_closes_when_ready_send_fails() -> None:
    connection = ScriptedConnection(
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0")),
        fail_send_call=1,
    )
    source = FakeSource([])

    SensorIpcSession(
        cast(socket.socket, connection),
        interface_resolver=_resolve_interface,
        source_factory=lambda interface: source,
    ).run()

    assert connection.send_calls == 1
    assert source.open_calls == 1
    assert source.receive_calls == 0
    assert source.close_calls == 1


def test_session_closes_if_client_disconnects_after_packet() -> None:
    connection = ScriptedConnection(
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0")),
        peek_results=[
            BlockingIOError("connected"),
            b"",
        ],
    )
    source = FakeSource([_packet()])

    SensorIpcSession(
        cast(socket.socket, connection),
        interface_resolver=_resolve_interface,
        source_factory=lambda interface: source,
    ).run()

    assert source.receive_calls == 1
    assert source.close_calls == 1

    assert len(connection.sent) == 1
    assert decode_sensor_ipc_message(connection.sent[0]) == SensorReadyMessage(interface="eth0")


def test_session_closes_when_packet_send_fails() -> None:
    connection = ScriptedConnection(
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0")),
        peek_results=[
            BlockingIOError("connected"),
            BlockingIOError("connected"),
        ],
        fail_send_call=2,
    )
    source = FakeSource([_packet()])

    SensorIpcSession(
        cast(socket.socket, connection),
        interface_resolver=_resolve_interface,
        source_factory=lambda interface: source,
    ).run()

    assert connection.send_calls == 2
    assert source.receive_calls == 1
    assert source.close_calls == 1


def test_session_treats_probe_os_error_as_disconnect() -> None:
    connection = ScriptedConnection(
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0")),
        peek_results=[
            OSError("connection probe failed"),
        ],
    )
    source = FakeSource([])

    SensorIpcSession(
        cast(socket.socket, connection),
        interface_resolver=_resolve_interface,
        source_factory=lambda interface: source,
    ).run()

    assert source.open_calls == 1
    assert source.receive_calls == 0
    assert source.close_calls == 1


def test_server_opens_real_unix_socket(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"
    server = UnixSensorServer(path)

    server.open()

    assert server.is_open
    assert server.socket_path == path
    assert path.exists()

    server.close()

    assert not server.is_open
    assert not path.exists()


def test_server_rejects_existing_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"
    path.write_text("do not replace me")

    with pytest.raises(
        SensorIpcServerError,
        match="already exists",
    ):
        UnixSensorServer(path).open()

    assert path.read_text() == "do not replace me"


def test_server_rejects_double_open(
    tmp_path: Path,
) -> None:
    server = UnixSensorServer(tmp_path / "sensor.sock")
    server.open()

    try:
        with pytest.raises(
            SensorIpcServerError,
            match="already open",
        ):
            server.open()
    finally:
        server.close()


def test_server_requires_open_before_serve(
    tmp_path: Path,
) -> None:
    server = UnixSensorServer(tmp_path / "sensor.sock")

    with pytest.raises(
        SensorIpcServerError,
        match="not open",
    ):
        server.serve_one()


def test_server_serves_real_unix_connection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"
    source = FakeSource([])

    server = UnixSensorServer(
        path,
        interface_resolver=lambda name: _interface(),
        source_factory=lambda interface: source,
    )
    server.open()

    thread = Thread(
        target=server.serve_one,
        daemon=True,
    )
    thread.start()

    client = socket.socket(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )

    try:
        client.connect(str(path))
        client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

        assert _recv_message(client) == (SensorReadyMessage(interface="eth0"))

        client.close()
        _join(thread)
        assert source.close_calls == 1
    finally:
        server.close()


def test_server_open_failure_without_created_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    listener = FakeUnixListener(
        bind_error=OSError("bind failed"),
    )

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: listener,
    )

    server = UnixSensorServer(path)

    with pytest.raises(
        SensorIpcServerError,
        match="could not open sensor IPC server",
    ):
        server.open()

    assert listener.closed
    assert not path.exists()
    assert not server.is_open


def test_server_open_failure_removes_created_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    listener = FakeUnixListener(
        create_path_on_bind=True,
        bind_error=OSError("bind failed"),
    )

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: listener,
    )

    server = UnixSensorServer(path)

    with pytest.raises(
        SensorIpcServerError,
        match="could not open sensor IPC server",
    ):
        server.open()

    assert listener.closed
    assert not path.exists()
    assert not server.is_open


def test_server_wraps_accept_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    listener = FakeUnixListener(
        accept_error=OSError("accept failed"),
    )

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: listener,
    )

    server = UnixSensorServer(path)
    server.open()

    try:
        with pytest.raises(
            SensorIpcServerError,
            match="could not accept sensor IPC client",
        ):
            server.serve_one()
    finally:
        server.close()


def test_server_wraps_listener_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    listener = FakeUnixListener(
        close_error=OSError("close failed"),
    )

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: listener,
    )

    server = UnixSensorServer(path)
    server.open()

    with pytest.raises(
        SensorIpcServerError,
        match="could not close sensor IPC server",
    ):
        server.close()

    assert listener.closed
    assert not server.is_open


def test_server_wraps_socket_path_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    server = UnixSensorServer(path)
    server.open()

    original_unlink = Path.unlink

    def fail_unlink(
        target: Path,
        missing_ok: bool = False,
    ) -> None:
        if target == path:
            raise OSError("unlink failed")

        original_unlink(
            target,
            missing_ok=missing_ok,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_unlink,
    )

    try:
        with pytest.raises(
            SensorIpcServerError,
            match="could not close sensor IPC server",
        ):
            server.close()
    finally:
        monkeypatch.undo()

        if path.exists():
            original_unlink(path)

    assert not server.is_open


def test_server_preserves_first_close_error_when_unlink_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor.sock"
    listener = FakeUnixListener(
        create_path_on_bind=True,
        close_error=OSError("listener close failed"),
    )

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *_args, **_kwargs: listener,
    )

    server = UnixSensorServer(path)
    server.open()

    original_unlink = Path.unlink

    def fail_unlink(
        target: Path,
        missing_ok: bool = False,
    ) -> None:
        if target == path:
            raise OSError("unlink failed")

        original_unlink(
            target,
            missing_ok=missing_ok,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_unlink,
    )

    try:
        with pytest.raises(
            SensorIpcServerError,
            match="could not close sensor IPC server",
        ) as failure:
            server.close()

        assert failure.value.__cause__ is not None
        assert str(failure.value.__cause__) == "listener close failed"
    finally:
        monkeypatch.undo()

        if path.exists():
            original_unlink(path)


def test_server_context_manager_cleans_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"

    with UnixSensorServer(path) as server:
        assert server.is_open
        assert path.exists()

    assert not server.is_open
    assert not path.exists()


def test_server_close_is_idempotent(
    tmp_path: Path,
) -> None:
    server = UnixSensorServer(tmp_path / "sensor.sock")

    server.close()
    server.open()
    server.close()
    server.close()

    assert not server.is_open


def test_session_contains_generic_send_os_error() -> None:
    class GenericSendFailureConnection(
        ScriptedConnection,
    ):
        def sendall(self, data: bytes) -> None:
            del data
            self.send_calls += 1
            raise OSError("client transport failed")

    connection = GenericSendFailureConnection(
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0"))
    )
    source = FakeSource([])

    SensorIpcSession(
        cast(socket.socket, connection),
        interface_resolver=_resolve_interface,
        source_factory=lambda interface: source,
    ).run()

    assert connection.send_calls == 1
    assert source.open_calls == 1
    assert source.receive_calls == 0
    assert source.close_calls == 1


def test_server_survives_capture_close_failure_and_serves_next_client(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"

    first_source = FakeSource(
        [],
        close_error=SensorCaptureError("capture cleanup failed"),
    )
    second_source = FakeSource([])

    sources = deque(
        [
            first_source,
            second_source,
        ]
    )

    server = UnixSensorServer(
        path,
        interface_resolver=lambda name: _interface(),
        source_factory=lambda interface: sources.popleft(),
    )
    server.open()

    try:
        for expected_source in (
            first_source,
            second_source,
        ):
            thread = Thread(
                target=server.serve_one,
                daemon=True,
            )
            thread.start()

            client = socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM,
            )

            try:
                client.connect(str(path))
                client.sendall(encode_sensor_ipc_message(SensorStartRequest(interface="eth0")))

                assert _recv_message(client) == (SensorReadyMessage(interface="eth0"))

                client.close()
                _join(thread)

                assert expected_source.open_calls == 1
                assert expected_source.close_calls == 1
                assert server.is_open
            finally:
                client.close()

        assert not sources
    finally:
        server.close()

    assert not server.is_open
    assert not path.exists()
