import socket

import pytest

from parallax.sensor import (
    CapturedEthernetFrame,
    CaptureInterface,
    LiveEthernetCapture,
    SensorCaptureError,
)


class FakeSocket:
    def __init__(self) -> None:
        self.bound_address: tuple[str, int] | None = None
        self.closed = False
        self.recv_sizes: list[int] = []
        self.recv_result = b"ethernet-frame"
        self.bind_error: OSError | None = None
        self.recv_error: OSError | None = None
        self.close_error: OSError | None = None

    def bind(self, address: tuple[str, int]) -> None:
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_address = address

    def recv(self, bufsize: int) -> bytes:
        self.recv_sizes.append(bufsize)
        if self.recv_error is not None:
            raise self.recv_error
        return self.recv_result

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _interface() -> CaptureInterface:
    return CaptureInterface(index=2, name="eth0")


def test_opens_and_binds_selected_interface() -> None:
    fake_socket = FakeSocket()

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )

    capture.open()

    assert capture.interface == _interface()
    assert capture.is_open
    assert fake_socket.bound_address == ("eth0", 0)

    capture.close()


def test_default_factory_opens_linux_af_packet_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_socket = FakeSocket()
    calls: list[tuple[int, int, int]] = []

    def fake_socket_factory(
        family: int,
        socket_type: int,
        protocol: int,
    ) -> FakeSocket:
        calls.append((family, socket_type, protocol))
        return fake_socket

    monkeypatch.setattr(socket, "socket", fake_socket_factory)

    capture = LiveEthernetCapture(_interface())
    capture.open()

    assert calls == [
        (
            socket.AF_PACKET,
            socket.SOCK_RAW,
            socket.htons(0x0003),
        )
    ]
    assert fake_socket.bound_address == ("eth0", 0)

    capture.close()


def test_rejects_open_when_already_open() -> None:
    fake_socket = FakeSocket()
    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )

    capture.open()

    with pytest.raises(
        SensorCaptureError,
        match="live capture socket is already open",
    ):
        capture.open()

    capture.close()


def test_wraps_permission_error_opening_capture_socket() -> None:
    def denied_factory() -> FakeSocket:
        raise PermissionError("operation not permitted")

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=denied_factory,
    )

    with pytest.raises(
        SensorCaptureError,
        match="permission denied opening Linux packet capture socket",
    ):
        capture.open()

    assert not capture.is_open


def test_wraps_operating_system_error_opening_capture_socket() -> None:
    def failing_factory() -> FakeSocket:
        raise OSError("socket failure")

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=failing_factory,
    )

    with pytest.raises(
        SensorCaptureError,
        match="could not open Linux packet capture socket",
    ):
        capture.open()

    assert not capture.is_open


def test_bind_failure_closes_unowned_socket() -> None:
    fake_socket = FakeSocket()
    fake_socket.bind_error = OSError("device disappeared")

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )

    with pytest.raises(
        SensorCaptureError,
        match="could not bind capture socket to interface eth0",
    ):
        capture.open()

    assert fake_socket.closed
    assert not capture.is_open


def test_receive_requires_open_socket() -> None:
    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=FakeSocket,
    )

    with pytest.raises(
        SensorCaptureError,
        match="live capture socket is not open",
    ):
        capture.receive()


def test_receives_timestamped_ethernet_frame() -> None:
    fake_socket = FakeSocket()

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
        clock=lambda: 123.5,
    )
    capture.open()

    assert capture.receive() == CapturedEthernetFrame(
        timestamp_seconds=123.5,
        data=b"ethernet-frame",
    )
    assert fake_socket.recv_sizes == [65_535]

    capture.close()


def test_wraps_receive_error() -> None:
    fake_socket = FakeSocket()
    fake_socket.recv_error = OSError("receive failed")

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )
    capture.open()

    with pytest.raises(
        SensorCaptureError,
        match="could not receive Ethernet frame",
    ):
        capture.receive()

    capture.close()


def test_close_is_idempotent() -> None:
    fake_socket = FakeSocket()

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )

    capture.close()
    capture.open()
    capture.close()
    capture.close()

    assert fake_socket.closed
    assert not capture.is_open


def test_wraps_close_error_and_releases_ownership() -> None:
    fake_socket = FakeSocket()
    fake_socket.close_error = OSError("close failed")

    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )
    capture.open()

    with pytest.raises(
        SensorCaptureError,
        match="could not close live capture socket",
    ):
        capture.close()

    assert not capture.is_open


def test_context_manager_opens_and_closes_capture() -> None:
    fake_socket = FakeSocket()
    capture = LiveEthernetCapture(
        _interface(),
        socket_factory=lambda: fake_socket,
    )

    with capture as active_capture:
        assert active_capture is capture
        assert capture.is_open

    assert fake_socket.closed
    assert not capture.is_open
