"""Capture-interface discovery contracts for live Parallax sensors."""

import socket
from dataclasses import dataclass


class SensorInterfaceError(RuntimeError):
    """Raised when a capture interface cannot be discovered or resolved."""


@dataclass(frozen=True, slots=True)
class CaptureInterface:
    """Immutable identity for one operating-system network interface."""

    index: int
    name: str


def list_capture_interfaces() -> tuple[CaptureInterface, ...]:
    """Return available network interfaces in deterministic index order."""
    try:
        entries = socket.if_nameindex()
    except OSError as error:
        raise SensorInterfaceError("could not enumerate capture interfaces") from error

    return tuple(CaptureInterface(index=index, name=name) for index, name in sorted(entries))


def resolve_capture_interface(name: str) -> CaptureInterface:
    """Resolve one explicitly selected interface by operating-system name."""
    if not name.strip():
        raise SensorInterfaceError("capture interface name must not be empty")

    try:
        index = socket.if_nametoindex(name)
    except OSError as error:
        raise SensorInterfaceError(f"capture interface not found: {name}") from error

    return CaptureInterface(index=index, name=name)
