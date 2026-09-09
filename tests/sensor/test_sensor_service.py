import sys
from pathlib import Path
from types import TracebackType
from typing import ClassVar

import pytest

import parallax.sensor.service as service
from parallax.sensor.ipc_client import (
    DEFAULT_SENSOR_IPC_SOCKET_PATH,
)


class FakeServer:
    created_paths: ClassVar[list[Path]] = []
    serve_calls: ClassVar[int] = 0
    entered: ClassVar[bool] = False
    exited: ClassVar[bool] = False

    def __init__(
        self,
        socket_path: str | Path,
    ) -> None:
        self.socket_path = Path(socket_path)
        type(self).created_paths.append(self.socket_path)

    def __enter__(self) -> "FakeServer":
        type(self).entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        type(self).exited = True

    def serve_one(self) -> None:
        type(self).serve_calls += 1
        raise KeyboardInterrupt


@pytest.fixture(autouse=True)
def reset_fake_server() -> None:
    FakeServer.created_paths.clear()
    FakeServer.serve_calls = 0
    FakeServer.entered = False
    FakeServer.exited = False


def _install_fake_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service,
        "UnixSensorServer",
        FakeServer,
    )


def test_run_uses_default_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_server(monkeypatch)

    service.run([])

    assert FakeServer.created_paths == [DEFAULT_SENSOR_IPC_SOCKET_PATH]
    assert FakeServer.serve_calls == 1
    assert FakeServer.entered
    assert FakeServer.exited


def test_run_accepts_socket_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_server(monkeypatch)

    service.run(
        [
            "--socket",
            "/tmp/custom-parallax.sock",
        ]
    )

    assert FakeServer.created_paths == [Path("/tmp/custom-parallax.sock")]


def test_main_accepts_explicit_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_server(monkeypatch)

    service.main(
        [
            "--socket",
            "/tmp/explicit.sock",
        ]
    )

    assert FakeServer.created_paths == [Path("/tmp/explicit.sock")]


def test_main_uses_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_server(monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallax-sensor",
            "--socket",
            "/tmp/process.sock",
        ],
    )

    service.main()

    assert FakeServer.created_paths == [Path("/tmp/process.sock")]
