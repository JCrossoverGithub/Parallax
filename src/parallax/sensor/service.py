"""Executable service entry point for the Parallax packet metadata sensor."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from parallax.sensor.ipc_client import (
    DEFAULT_SENSOR_IPC_SOCKET_PATH,
)
from parallax.sensor.ipc_server import UnixSensorServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallax-sensor",
        description=("Run the local least-privilege Parallax packet metadata sensor."),
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=DEFAULT_SENSOR_IPC_SOCKET_PATH,
        help=(f"Unix-domain socket path (default: {DEFAULT_SENSOR_IPC_SOCKET_PATH})"),
    )
    return parser


def run(argv: Sequence[str]) -> None:
    """Run the sensor until interrupted."""
    arguments = _build_parser().parse_args(argv)

    with UnixSensorServer(arguments.socket) as server:
        try:
            while True:
                server.serve_one()
        except KeyboardInterrupt:
            return


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run using process arguments unless explicit arguments are supplied."""
    run(sys.argv[1:] if argv is None else argv)
