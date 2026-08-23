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
from parallax.features import (
    RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256,
    RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SHA256,
    RELEASE_COMPATIBLE_WINDOW_SHA256,
    ByteTotalPolicy,
    FeatureCalculationConfig,
    export_capture_split_manifest,
    export_vnat_features,
)
from parallax.modeling import (
    export_baseline_validation_report,
    export_prototype_ood_calibration,
    export_prototype_validation_report,
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

    feature_parser = dataset_commands.add_parser(
        "extract-features",
        help="calculate versioned VNAT features from window Parquet",
    )
    feature_parser.add_argument("source", help="path to a vnat-window-1 Parquet artifact")
    feature_parser.add_argument("output", help="destination .parquet path")
    feature_parser.add_argument(
        "--expected-sha256",
        default=RELEASE_COMPATIBLE_WINDOW_SHA256,
        help="trusted window-artifact SHA-256 checked before processing",
    )
    feature_parser.add_argument(
        "--byte-total-policy",
        type=ByteTotalPolicy,
        choices=tuple(ByteTotalPolicy),
        default=ByteTotalPolicy.RELEASE_COMPATIBLE,
        help="directional byte-total treatment (default: %(default)s)",
    )
    feature_parser.add_argument(
        "--window-seconds",
        type=float,
        default=WINDOW_SECONDS,
        help="window duration in seconds (default: %(default)s)",
    )
    feature_parser.add_argument(
        "--time-bin-seconds",
        type=float,
        default=0.01,
        help="wavelet signal bin duration in seconds (default: %(default)s)",
    )
    feature_parser.set_defaults(handler=_extract_features)

    split_parser = dataset_commands.add_parser(
        "split-features",
        help="create an immutable capture-grouped split manifest",
    )
    split_parser.add_argument("source", help="path to a vnat-feature-artifact-1 Parquet file")
    split_parser.add_argument("output", help="destination .json manifest path")
    split_parser.add_argument(
        "--expected-sha256",
        default=RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SHA256,
        help="trusted feature-artifact SHA-256 checked before processing",
    )
    split_parser.set_defaults(handler=_split_features)

    model_parser = commands.add_parser("model", help="run leakage-resistant model experiments")
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    baseline_parser = model_commands.add_parser(
        "validate-baselines",
        help="fit training-only baselines and publish validation metrics",
    )
    baseline_parser.add_argument("features", help="path to a vnat-feature-artifact-1 Parquet file")
    baseline_parser.add_argument(
        "split_manifest", help="path to its trusted capture-split manifest"
    )
    baseline_parser.add_argument("output", help="destination .json validation report path")
    baseline_parser.add_argument(
        "--expected-manifest-sha256",
        default=RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256,
        help="trusted split-manifest SHA-256 checked before processing",
    )
    baseline_parser.set_defaults(handler=_validate_baselines)

    prototype_parser = model_commands.add_parser(
        "validate-prototype",
        help="train a prototype model and publish validation-only artifacts",
    )
    prototype_parser.add_argument("features", help="path to a vnat-feature-artifact-1 Parquet file")
    prototype_parser.add_argument(
        "split_manifest", help="path to its trusted capture-split manifest"
    )
    prototype_parser.add_argument("model_bundle", help="destination .json model bundle path")
    prototype_parser.add_argument("output", help="destination .json validation report path")
    prototype_parser.add_argument(
        "--expected-manifest-sha256",
        default=RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256,
        help="trusted split-manifest SHA-256 checked before processing",
    )
    prototype_parser.set_defaults(handler=_validate_prototype)

    calibration_parser = model_commands.add_parser(
        "calibrate-prototype",
        help="fit and publish OOD calibration for a frozen prototype model",
    )
    calibration_parser.add_argument(
        "features", help="path to a vnat-feature-artifact-1 Parquet file"
    )
    calibration_parser.add_argument(
        "split_manifest", help="path to its trusted capture-split manifest"
    )
    calibration_parser.add_argument("model_bundle", help="path to the frozen .json model bundle")
    calibration_parser.add_argument("output", help="destination .json calibration artifact path")
    calibration_parser.add_argument(
        "--expected-manifest-sha256",
        default=RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256,
        help="trusted split-manifest SHA-256 checked before processing",
    )
    calibration_parser.add_argument(
        "--expected-model-bundle-sha256",
        required=True,
        help="trusted frozen model-bundle SHA-256 checked before calibration",
    )
    calibration_parser.set_defaults(handler=_calibrate_prototype)

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


def _extract_features(arguments: argparse.Namespace) -> None:
    config = FeatureCalculationConfig(
        window_seconds=arguments.window_seconds,
        time_bin_seconds=arguments.time_bin_seconds,
        byte_total_policy=arguments.byte_total_policy,
    )
    report = export_vnat_features(
        arguments.source,
        arguments.output,
        expected_sha256=arguments.expected_sha256,
        config=config,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _split_features(arguments: argparse.Namespace) -> None:
    report = export_capture_split_manifest(
        arguments.source,
        arguments.output,
        expected_sha256=arguments.expected_sha256,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _validate_baselines(arguments: argparse.Namespace) -> None:
    report = export_baseline_validation_report(
        arguments.features,
        arguments.split_manifest,
        arguments.output,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _validate_prototype(arguments: argparse.Namespace) -> None:
    report = export_prototype_validation_report(
        arguments.features,
        arguments.split_manifest,
        arguments.model_bundle,
        arguments.output,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def _calibrate_prototype(arguments: argparse.Namespace) -> None:
    artifact = export_prototype_ood_calibration(
        arguments.features,
        arguments.split_manifest,
        arguments.model_bundle,
        arguments.output,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        expected_model_bundle_sha256=arguments.expected_model_bundle_sha256,
    )
    print(json.dumps(artifact.as_dict(), indent=2, sort_keys=True))


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
