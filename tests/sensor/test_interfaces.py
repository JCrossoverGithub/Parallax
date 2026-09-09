import socket

import pytest

from parallax.sensor import (
    CaptureInterface,
    SensorInterfaceError,
    list_capture_interfaces,
    resolve_capture_interface,
)


def test_list_capture_interfaces_returns_deterministic_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_if_nameindex() -> list[tuple[int, str]]:
        return [
            (7, "eth1"),
            (1, "lo"),
            (3, "eth0"),
        ]

    monkeypatch.setattr(socket, "if_nameindex", fake_if_nameindex)

    assert list_capture_interfaces() == (
        CaptureInterface(index=1, name="lo"),
        CaptureInterface(index=3, name="eth0"),
        CaptureInterface(index=7, name="eth1"),
    )


def test_list_capture_interfaces_allows_no_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_if_nameindex() -> list[tuple[int, str]]:
        return []

    monkeypatch.setattr(socket, "if_nameindex", fake_if_nameindex)

    assert list_capture_interfaces() == ()


def test_list_capture_interfaces_wraps_operating_system_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_if_nameindex() -> list[tuple[int, str]]:
        raise OSError("interface enumeration failed")

    monkeypatch.setattr(socket, "if_nameindex", fake_if_nameindex)

    with pytest.raises(
        SensorInterfaceError,
        match="could not enumerate capture interfaces",
    ):
        list_capture_interfaces()


def test_resolve_capture_interface_returns_selected_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_if_nametoindex(name: str) -> int:
        assert name == "eth0"
        return 3

    monkeypatch.setattr(socket, "if_nametoindex", fake_if_nametoindex)

    assert resolve_capture_interface("eth0") == CaptureInterface(
        index=3,
        name="eth0",
    )


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_resolve_capture_interface_rejects_blank_name(name: str) -> None:
    with pytest.raises(
        SensorInterfaceError,
        match="capture interface name must not be empty",
    ):
        resolve_capture_interface(name)


def test_resolve_capture_interface_wraps_missing_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_if_nametoindex(name: str) -> int:
        assert name == "missing0"
        raise OSError("no such device")

    monkeypatch.setattr(socket, "if_nametoindex", fake_if_nametoindex)

    with pytest.raises(
        SensorInterfaceError,
        match="capture interface not found: missing0",
    ):
        resolve_capture_interface("missing0")
