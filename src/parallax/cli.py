"""Command-line entry point for Parallax."""

import argparse
import json
import sys
from collections.abc import Sequence

from parallax import __version__
from parallax.data import (
    MIN_PACKETS_PER_WINDOW,
    RAW_RELEASE_1_SHA256,
    WINDOW_SECONDS,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    export_vnat_windows,
    inspect_vnat_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parallax")
    commands = parser.add_subparsers(dest="command")

    dataset_parser = commands.add_parser("dataset", help="work with dataset artifacts")
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
    inspect_parser.set_defaults(handler=_inspect_dataset)

    extract_parser = dataset_commands.add_parser(
        "extract-windows",
        help="export deterministic VNAT windows to Parquet",
    )
    extract_parser.add_argument("source", help="path to VNAT_Dataframe_release_1.h5")
    extract_parser.add_argument("output", help="destination .parquet path")
    extract_parser.add_argument(
        "--expected-sha256",
        default=RAW_RELEASE_1_SHA256,
        help="trusted SHA-256 checked before deserialization",
    )
    extract_parser.add_argument(
        "--threshold-policy",
        type=WindowThresholdPolicy,
        choices=tuple(WindowThresholdPolicy),
        default=WindowThresholdPolicy.RELEASE_COMPATIBLE,
        help="packet-threshold interpretation (default: %(default)s)",
    )
    extract_parser.add_argument(
        "--window-seconds",
        type=float,
        default=WINDOW_SECONDS,
        help="window duration in seconds (default: %(default)s)",
    )
    extract_parser.add_argument(
        "--minimum-packets",
        type=int,
        default=MIN_PACKETS_PER_WINDOW,
        help="packet threshold used by the selected policy (default: %(default)s)",
    )
    extract_parser.set_defaults(handler=_extract_windows)

    return parser


def _inspect_dataset(arguments: argparse.Namespace) -> None:
    report = inspect_vnat_file(arguments.path, expected_sha256=arguments.expected_sha256)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _extract_windows(arguments: argparse.Namespace) -> None:
    config = WindowExtractionConfig(
        window_seconds=arguments.window_seconds,
        minimum_packets=arguments.minimum_packets,
        threshold_policy=arguments.threshold_policy,
    )
    report = export_vnat_windows(
        arguments.source,
        arguments.output,
        expected_sha256=arguments.expected_sha256,
        config=config,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def run(argv: Sequence[str]) -> None:
    """Run the Parallax CLI with explicit arguments."""
    if not argv:
        print(f"Parallax {__version__}")
        return

    arguments = _build_parser().parse_args(argv)
    arguments.handler(arguments)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Parallax using process arguments unless arguments are supplied."""
    run(sys.argv[1:] if argv is None else argv)
