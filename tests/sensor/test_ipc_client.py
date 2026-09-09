import socket
from collections import deque
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from parallax.data import PacketMetadata
from parallax.sensor import CaptureInterface
from parallax.sensor.ipc_client import (
    DEFAULT_SENSOR_IPC_SOCKET_PATH,
    SensorIpcClientError,
    SensorIpcClientSocket,
    SensorIpcPacketSource,
)
from parallax.sensor.ipc_protocol import (
    SENSOR_IPC_MAX_MESSAGE_BYTES,
    SensorErrorMessage,
    SensorPacketMessage,
    SensorReadyMessage,
    SensorStartRequest,
    decode_sensor_ipc_message,
    encode_sensor_ipc_message,
)
from parallax.sensor.ipc_server import (
    SensorPacketSource,
    UnixSensorServer,
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


class ScriptedSocket:
    def __init__(
        self,
        recv_results: list[bytes | BaseException],
        *,
        connect_error: OSError | None = None,
        send_error: OSError | None = None,
        close_error: OSError | None = None,
    ) -> None:
        self.recv_results = deque(recv_results)
        self.connect_error = connect_error
        self.send_error = send_error
        self.close_error = close_error
        self.timeouts: list[float | None] = []
        self.connected_path: str | None = None
        self.sent: list[bytes] = []
        self.closed = False

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        self.timeouts.append(value)

    def connect(
        self,
        address: str,
    ) -> None:
        if self.connect_error is not None:
            raise self.connect_error

        self.connected_path = address

    def sendall(
        self,
        data: bytes,
    ) -> None:
        if self.send_error is not None:
            raise self.send_error

        self.sent.append(data)

    def recv(
        self,
        bufsize: int,
    ) -> bytes:
        if not self.recv_results:
            raise AssertionError("scripted socket recv queue exhausted")

        result = self.recv_results.popleft()

        if isinstance(
            result,
            BaseException,
        ):
            raise result

        if len(result) <= bufsize:
            return result

        head = result[:bufsize]
        self.recv_results.appendleft(result[bufsize:])
        return head

    def close(self) -> None:
        self.closed = True

        if self.close_error is not None:
            raise self.close_error


def _source(
    fake_socket: ScriptedSocket,
    *,
    socket_path: Path = Path("/tmp/parallax-test.sock"),
) -> SensorIpcPacketSource:
    return SensorIpcPacketSource(
        socket_path,
        "eth0",
        socket_factory=lambda: cast(
            SensorIpcClientSocket,
            fake_socket,
        ),
    )


def _ready() -> bytes:
    return encode_sensor_ipc_message(SensorReadyMessage(interface="eth0"))


def _packet_message() -> bytes:
    return encode_sensor_ipc_message(SensorPacketMessage(packet=_packet()))


def test_default_socket_path_is_system_runtime_path() -> None:
    assert Path("/run/parallax/sensor.sock") == DEFAULT_SENSOR_IPC_SOCKET_PATH


def test_open_connects_sends_start_and_sets_poll_timeout() -> None:
    fake_socket = ScriptedSocket([_ready()])
    source = _source(fake_socket)

    source.open()

    assert source.is_open
    assert source.interface == "eth0"
    assert source.socket_path == Path("/tmp/parallax-test.sock")
    assert fake_socket.connected_path == ("/tmp/parallax-test.sock")
    assert fake_socket.timeouts == [
        2.0,
        0.25,
    ]

    assert len(fake_socket.sent) == 1
    assert decode_sensor_ipc_message(fake_socket.sent[0]) == SensorStartRequest(interface="eth0")

    source.close()

    assert fake_socket.closed
    assert not source.is_open


def test_open_accepts_fragmented_ready_message() -> None:
    ready = _ready()
    split = len(ready) // 2

    source = _source(
        ScriptedSocket(
            [
                ready[:split],
                ready[split:],
            ]
        )
    )

    source.open()
    assert source.is_open
    source.close()


def test_receive_preserves_combined_message_remainder() -> None:
    source = _source(ScriptedSocket([_ready() + _packet_message()]))

    source.open()

    assert source.receive() == _packet()

    source.close()


def test_receive_returns_none_after_quiet_poll() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                TimeoutError("quiet poll"),
            ]
        )
    )
    source.open()

    assert source.receive() is None

    source.close()


def test_partial_message_survives_quiet_poll() -> None:
    packet = _packet_message()
    split = len(packet) // 2

    source = _source(
        ScriptedSocket(
            [
                _ready(),
                packet[:split],
                TimeoutError("quiet poll"),
                packet[split:],
            ]
        )
    )
    source.open()

    assert source.receive() is None
    assert source.receive() == _packet()

    source.close()


def test_open_rejects_sensor_error() -> None:
    fake_socket = ScriptedSocket(
        [
            encode_sensor_ipc_message(
                SensorErrorMessage(
                    code="capture_error",
                    message="permission denied",
                )
            )
        ]
    )
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="permission denied",
    ) as failure:
        source.open()

    assert failure.value.code == ("capture_error")
    assert fake_socket.closed
    assert not source.is_open


def test_receive_raises_structured_sensor_error() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                encode_sensor_ipc_message(
                    SensorErrorMessage(
                        code="capture_error",
                        message="capture failed",
                    )
                ),
            ]
        )
    )
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="capture failed",
    ) as failure:
        source.receive()

    assert failure.value.code == ("capture_error")

    source.close()


def test_open_rejects_unexpected_handshake_message() -> None:
    fake_socket = ScriptedSocket([_packet_message()])
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="did not send a ready message",
    ):
        source.open()

    assert fake_socket.closed


def test_open_rejects_mismatched_ready_interface() -> None:
    fake_socket = ScriptedSocket([encode_sensor_ipc_message(SensorReadyMessage(interface="eth1"))])
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="ready interface does not match",
    ):
        source.open()

    assert fake_socket.closed


def test_receive_rejects_unexpected_ready_message() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                _ready(),
            ]
        )
    )
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="unexpected message",
    ):
        source.receive()

    source.close()


def test_receive_requires_open_source() -> None:
    source = _source(ScriptedSocket([]))

    with pytest.raises(
        SensorIpcClientError,
        match="is not open",
    ):
        source.receive()


def test_open_rejects_double_open() -> None:
    source = _source(ScriptedSocket([_ready()]))
    source.open()

    try:
        with pytest.raises(
            SensorIpcClientError,
            match="already open",
        ):
            source.open()
    finally:
        source.close()


@pytest.mark.parametrize(
    "timeout",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_handshake_timeout(
    timeout: float,
) -> None:
    with pytest.raises(
        SensorIpcClientError,
        match="handshake timeout must be finite and positive",
    ):
        SensorIpcPacketSource(
            "/tmp/sensor.sock",
            "eth0",
            handshake_timeout_seconds=timeout,
        )


@pytest.mark.parametrize(
    "timeout",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_poll_timeout(
    timeout: float,
) -> None:
    with pytest.raises(
        SensorIpcClientError,
        match="poll timeout must be finite and positive",
    ):
        SensorIpcPacketSource(
            "/tmp/sensor.sock",
            "eth0",
            poll_timeout_seconds=timeout,
        )


def test_rejects_blank_interface() -> None:
    with pytest.raises(
        SensorIpcClientError,
        match="interface must not be empty",
    ):
        SensorIpcPacketSource(
            "/tmp/sensor.sock",
            " ",
        )


def test_wraps_connect_failure() -> None:
    fake_socket = ScriptedSocket(
        [],
        connect_error=OSError("connection refused"),
    )
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="could not open sensor IPC packet source",
    ):
        source.open()

    assert fake_socket.closed


def test_wraps_start_send_failure() -> None:
    fake_socket = ScriptedSocket(
        [],
        send_error=OSError("send failed"),
    )
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="could not open sensor IPC packet source",
    ):
        source.open()

    assert fake_socket.closed


def test_wraps_handshake_timeout() -> None:
    fake_socket = ScriptedSocket([TimeoutError("handshake timeout")])
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="timed out waiting for sensor IPC response",
    ):
        source.open()

    assert fake_socket.closed


def test_wraps_handshake_protocol_failure() -> None:
    fake_socket = ScriptedSocket([b"{not-json}\n"])
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="sent an invalid message",
    ):
        source.open()

    assert fake_socket.closed


def test_wraps_receive_operating_system_error() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                OSError("recv failed"),
            ]
        )
    )
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="could not receive sensor IPC message",
    ):
        source.receive()

    source.close()


def test_reports_server_disconnect() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                b"",
            ]
        )
    )
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="server disconnected",
    ):
        source.receive()

    source.close()


def test_rejects_oversized_unframed_message() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                b"x" * SENSOR_IPC_MAX_MESSAGE_BYTES,
            ]
        )
    )
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="sent an invalid message",
    ):
        source.receive()

    source.close()


def test_close_is_idempotent() -> None:
    fake_socket = ScriptedSocket([_ready()])
    source = _source(fake_socket)

    source.close()
    source.open()
    source.close()
    source.close()

    assert fake_socket.closed
    assert not source.is_open


def test_wraps_close_failure_and_releases_state() -> None:
    fake_socket = ScriptedSocket(
        [_ready()],
        close_error=OSError("close failed"),
    )
    source = _source(fake_socket)
    source.open()

    with pytest.raises(
        SensorIpcClientError,
        match="could not close sensor IPC packet source",
    ):
        source.close()

    assert not source.is_open


def test_failed_open_ignores_cleanup_close_error() -> None:
    fake_socket = ScriptedSocket(
        [],
        connect_error=OSError("connection refused"),
        close_error=OSError("cleanup close failed"),
    )
    source = _source(fake_socket)

    with pytest.raises(
        SensorIpcClientError,
        match="could not open sensor IPC packet source",
    ):
        source.open()

    assert fake_socket.closed


class RealServerSource:
    def __init__(self) -> None:
        self.packet_sent = False
        self.closed = False

    def open(self) -> None:
        pass

    def receive(
        self,
    ) -> PacketMetadata | None:
        if self.packet_sent:
            return None

        self.packet_sent = True
        return _packet()

    def close(self) -> None:
        self.closed = True


def test_client_and_server_interoperate_over_real_unix_socket(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensor.sock"
    server_source = RealServerSource()

    server = UnixSensorServer(
        path,
        interface_resolver=lambda name: CaptureInterface(
            index=2,
            name=name,
        ),
        source_factory=lambda interface: cast(
            SensorPacketSource,
            server_source,
        ),
    )
    server.open()

    thread = Thread(
        target=server.serve_one,
        daemon=True,
    )
    thread.start()

    source = SensorIpcPacketSource(
        path,
        "eth0",
        handshake_timeout_seconds=1.0,
        poll_timeout_seconds=0.05,
    )

    try:
        source.open()
        assert source.receive() == _packet()
        source.close()

        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert server_source.closed
    finally:
        source.close()
        server.close()


def test_default_factory_creates_unix_stream_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[int, int]] = []
    fake_socket = ScriptedSocket([_ready()])

    def create_socket(
        family: int,
        socket_type: int,
    ) -> socket.socket:
        created.append(
            (
                family,
                socket_type,
            )
        )
        return cast(
            socket.socket,
            fake_socket,
        )

    monkeypatch.setattr(
        socket,
        "socket",
        create_socket,
    )

    source = SensorIpcPacketSource(
        "/tmp/sensor.sock",
        "eth0",
    )
    source.open()
    source.close()

    assert created == [
        (
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
    ]


def test_connect_failure_has_unavailable_code() -> None:
    source = _source(
        ScriptedSocket(
            [],
            connect_error=OSError("connection refused"),
        )
    )

    with pytest.raises(
        SensorIpcClientError,
    ) as failure:
        source.open()

    assert failure.value.code == "sensor_unavailable"


def test_handshake_timeout_has_timeout_code() -> None:
    source = _source(ScriptedSocket([TimeoutError("handshake timeout")]))

    with pytest.raises(
        SensorIpcClientError,
    ) as failure:
        source.open()

    assert failure.value.code == "sensor_timeout"


def test_invalid_handshake_has_protocol_code() -> None:
    source = _source(
        ScriptedSocket(
            [
                b"{not-json}\n",
            ]
        )
    )

    with pytest.raises(
        SensorIpcClientError,
    ) as failure:
        source.open()

    assert failure.value.code == "sensor_protocol_error"


def test_disconnect_has_disconnect_code() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                b"",
            ]
        )
    )

    source.open()

    try:
        with pytest.raises(
            SensorIpcClientError,
        ) as failure:
            source.receive()

        assert failure.value.code == "sensor_disconnected"
    finally:
        source.close()


def test_receive_transport_failure_has_ipc_code() -> None:
    source = _source(
        ScriptedSocket(
            [
                _ready(),
                OSError("recv failed"),
            ]
        )
    )

    source.open()

    try:
        with pytest.raises(
            SensorIpcClientError,
        ) as failure:
            source.receive()

        assert failure.value.code == "sensor_ipc_error"
    finally:
        source.close()


def test_invalid_client_configuration_has_configuration_code() -> None:
    with pytest.raises(
        SensorIpcClientError,
    ) as failure:
        SensorIpcPacketSource(
            "/tmp/sensor.sock",
            " ",
        )

    assert failure.value.code == "sensor_configuration_error"


def test_invalid_client_state_has_state_code() -> None:
    source = _source(ScriptedSocket([]))

    with pytest.raises(
        SensorIpcClientError,
    ) as failure:
        source.receive()

    assert failure.value.code == "sensor_state_error"


@pytest.mark.parametrize(
    "code",
    [
        "capture_error",
        "interface_error",
        "protocol_error",
        "invalid_request",
    ],
)
def test_preserves_server_originated_error_codes(
    code: str,
) -> None:
    source = _source(
        ScriptedSocket(
            [
                encode_sensor_ipc_message(
                    SensorErrorMessage(
                        code=code,
                        message="sensor reported failure",
                    )
                )
            ]
        )
    )

    with pytest.raises(
        SensorIpcClientError,
        match="sensor reported failure",
    ) as failure:
        source.open()

    assert failure.value.code == code
