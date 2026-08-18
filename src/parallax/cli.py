"""Command-line entry point for Parallax."""

import argparse
import json
import sys
from collections.abc import Sequence

from parallax import __version__
from parallax.data import RAW_RELEASE_1_SHA256, inspect_vnat_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallax")
    commands = parser.add_subparsers(dest="command")

    dataset_parser = commands.add_parser("dataset", help="inspect dataset artifacts")
    dataset_commands = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    inspect_parser = dataset_commands.add_parser(
        "inspect", help="validate and summarize a raw VNAT HDF5 file"
    )
    inspect_parser.add_argument("path", help="path to VNAT_Dataframe_release_1.h5")
    inspect_parser.add_argument(
        "--expected-sha256",
        default=RAW_RELEASE_1_SHA256,
        help="trusted SHA-256 checked before deserialization",
    )

    return parser


def run(argv: Sequence[str]) -> None:
    """Run the Parallax CLI with explicit arguments."""
    if not argv:
        print(f"Parallax {__version__}")
        return

    arguments = _build_parser().parse_args(argv)
    report = inspect_vnat_file(arguments.path, expected_sha256=arguments.expected_sha256)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    """Run Parallax using process arguments unless arguments are supplied."""
    run(sys.argv[1:] if argv is None else argv)
