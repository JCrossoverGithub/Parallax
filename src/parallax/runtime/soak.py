"""Synthetic sustained-load validation for the live runtime."""

import argparse
import json
import os
import resource
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import cast

from parallax.data import (
    PacketMetadata,
    RuntimeFlowCapacityError,
    RuntimeFlowTrackerConfig,
    WindowExtractionConfig,
)
from parallax.runtime.events import RuntimePredictionEvent
from parallax.runtime.live import (
    LiveRuntimeSummary,
    run_live_packet_predictions,
)
from parallax.runtime.pipeline import (
    PacketPredictionPipeline,
    RuntimePipelineStats,
    RuntimeScorer,
)

LIVE_SOAK_REPORT_SCHEMA_VERSION = "parallax-live-soak-1"
_SYNTHETIC_WINDOW_SECONDS = 10.0


class LiveSoakError(ValueError):
    """Raised when a synthetic live soak configuration is invalid."""


class LiveSoakProfile(StrEnum):
    """Traffic-shape profiles supported by the synthetic soak harness."""

    STEADY = "steady"
    STALE_CHURN = "stale-churn"
    CAPACITY = "capacity"


@dataclass(frozen=True, slots=True)
class LiveSoakConfig:
    """Configuration for one deterministic synthetic live soak."""

    profile: LiveSoakProfile = LiveSoakProfile.STEADY
    packets: int = 100_000
    flows: int = 1_024
    max_tracked_flows: int = 4_096
    stale_after_seconds: float = 120.0
    timestamp_step_seconds: float = 0.001

    def __post_init__(self) -> None:
        if self.packets < 1:
            raise LiveSoakError("soak packet count must be positive")

        if self.flows < 1:
            raise LiveSoakError("soak flow count must be positive")

        if self.max_tracked_flows < 1:
            raise LiveSoakError("maximum tracked flows must be positive")

        if not isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0.0:
            raise LiveSoakError("stale flow timeout must be finite and positive")

        if not isfinite(self.timestamp_step_seconds) or self.timestamp_step_seconds <= 0.0:
            raise LiveSoakError("timestamp step must be finite and positive")

        if self.stale_after_seconds < _SYNTHETIC_WINDOW_SECONDS:
            raise LiveSoakError(
                f"stale flow timeout must be at least {_SYNTHETIC_WINDOW_SECONDS:g} seconds"
            )

        if (
            self.profile
            in {
                LiveSoakProfile.STEADY,
                LiveSoakProfile.STALE_CHURN,
            }
            and self.flows > self.max_tracked_flows
        ):
            raise LiveSoakError(
                "non-capacity soak flow count must not exceed maximum tracked flows"
            )

        if (
            self.profile is LiveSoakProfile.STEADY
            and self.flows * self.timestamp_step_seconds >= self.stale_after_seconds
        ):
            raise LiveSoakError("steady profile must revisit each flow before the stale timeout")

        if self.profile is LiveSoakProfile.STALE_CHURN and self.packets <= self.flows:
            raise LiveSoakError("stale-churn profile requires more packets than flows")

        if self.profile is LiveSoakProfile.CAPACITY and self.flows <= self.max_tracked_flows:
            raise LiveSoakError("capacity profile requires more flows than maximum tracked flows")

        if self.profile is LiveSoakProfile.CAPACITY and self.packets <= self.max_tracked_flows:
            raise LiveSoakError(
                "capacity profile requires enough packets to exceed maximum tracked flows"
            )

    def as_dict(self) -> dict[str, object]:
        """Return the stable report configuration."""

        return {
            "profile": self.profile.value,
            "packets": self.packets,
            "flows": self.flows,
            "max_tracked_flows": self.max_tracked_flows,
            "stale_after_seconds": self.stale_after_seconds,
            "timestamp_step_seconds": self.timestamp_step_seconds,
            "synthetic_window_seconds": (_SYNTHETIC_WINDOW_SECONDS),
        }


class _SyntheticPacketSource:
    """Generate deterministic packet metadata without retaining packets."""

    __slots__ = (
        "_closed",
        "_config",
        "_emitted",
    )

    def __init__(
        self,
        config: LiveSoakConfig,
    ) -> None:
        self._config = config
        self._emitted = 0
        self._closed = False

    @property
    def emitted(self) -> int:
        """Return the number of packets emitted."""

        return self._emitted

    @property
    def closed(self) -> bool:
        """Return whether the source was closed."""

        return self._closed

    def open(self) -> None:
        """Open the synthetic source."""

    def receive(self) -> PacketMetadata:
        """Return the next deterministic packet."""

        packet = _synthetic_packet(
            self._config,
            self._emitted,
        )
        self._emitted += 1
        return packet

    def close(self) -> None:
        """Close the synthetic source."""

        self._closed = True


@dataclass(frozen=True, slots=True)
class LiveSoakReport:
    """Recorded result from one synthetic live soak."""

    config: LiveSoakConfig
    outcome: str
    source_packets_emitted: int
    source_closed: bool
    wall_elapsed_seconds: float
    runtime_summary: LiveRuntimeSummary | None
    pipeline_stats: RuntimePipelineStats
    peak_rss_before_kib: int
    peak_rss_after_kib: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible performance report."""

        runtime = None if self.runtime_summary is None else asdict(self.runtime_summary)

        return {
            "schema_version": LIVE_SOAK_REPORT_SCHEMA_VERSION,
            "configuration": self.config.as_dict(),
            "outcome": self.outcome,
            "measurement": {
                "source_packets_emitted": (self.source_packets_emitted),
                "source_closed": self.source_closed,
                "wall_elapsed_seconds": (self.wall_elapsed_seconds),
            },
            "runtime": runtime,
            "pipeline": asdict(self.pipeline_stats),
            "memory": {
                "peak_rss_before_kib": (self.peak_rss_before_kib),
                "peak_rss_after_kib": (self.peak_rss_after_kib),
                "peak_rss_growth_kib": max(
                    0,
                    self.peak_rss_after_kib - self.peak_rss_before_kib,
                ),
                "metric": ("resource.RUSAGE_SELF.ru_maxrss (Linux KiB)"),
            },
            "interpretation_boundary": {
                "synthetic_packet_metadata": True,
                "packet_payload_processed": False,
                "frozen_model_scoring_exercised": False,
                "accuracy_or_ood_evidence": False,
                "process_peak_rss_includes_setup_overhead": True,
            },
        }


def run_live_soak(
    config: LiveSoakConfig,
    *,
    _memory_probe: Callable[[], int] | None = None,
    _wall_clock: Callable[[], float] | None = None,
) -> LiveSoakReport:
    """Run one deterministic synthetic sustained-load profile."""

    memory_probe = _peak_rss_kib if _memory_probe is None else _memory_probe
    wall_clock = perf_counter if _wall_clock is None else _wall_clock

    source = _SyntheticPacketSource(config)

    pipeline = PacketPredictionPipeline(
        run_id="synthetic-live-soak",
        capture_id=(f"synthetic:{config.profile.value}"),
        scorer=cast(RuntimeScorer, object()),
        window_config=WindowExtractionConfig(
            window_seconds=_SYNTHETIC_WINDOW_SECONDS,
            minimum_packets=config.packets + 1,
        ),
        flow_config=RuntimeFlowTrackerConfig(
            stale_after_seconds=(config.stale_after_seconds),
            max_tracked_flows=(config.max_tracked_flows),
        ),
    )

    events: list[RuntimePredictionEvent] = []

    peak_before = memory_probe()
    started_at = wall_clock()

    summary: LiveRuntimeSummary | None

    if config.profile is LiveSoakProfile.CAPACITY:
        try:
            run_live_packet_predictions(
                source,
                pipeline=pipeline,
                packet_limit=config.packets,
                handle_event=events.append,
            )
        except RuntimeFlowCapacityError:
            summary = None
            outcome = "capacity_exceeded"
        else:
            raise LiveSoakError("capacity profile completed without reaching flow capacity")
    else:
        summary = run_live_packet_predictions(
            source,
            pipeline=pipeline,
            packet_limit=config.packets,
            handle_event=events.append,
        )
        outcome = "completed"

    wall_elapsed_seconds = wall_clock() - started_at
    peak_after = memory_probe()

    return LiveSoakReport(
        config=config,
        outcome=outcome,
        source_packets_emitted=source.emitted,
        source_closed=source.closed,
        wall_elapsed_seconds=wall_elapsed_seconds,
        runtime_summary=summary,
        pipeline_stats=pipeline.stats,
        peak_rss_before_kib=peak_before,
        peak_rss_after_kib=peak_after,
    )


def write_live_soak_report(
    report: LiveSoakReport,
    output: str | Path,
) -> Path:
    """Atomically write one synthetic soak report."""

    destination = Path(output)

    if destination.suffix.casefold() != ".json":
        raise LiveSoakError("live soak report must use the .json extension")

    encoded = (
        json.dumps(
            report.as_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix=".parallax-live-soak-",
        dir=destination.parent,
    ) as temporary:
        temporary_output = Path(temporary) / destination.name
        temporary_output.write_bytes(encoded)
        os.replace(
            temporary_output,
            destination,
        )

    return destination


def _synthetic_packet(
    config: LiveSoakConfig,
    index: int,
) -> PacketMetadata:
    if config.profile is LiveSoakProfile.STEADY:
        flow_index = index % config.flows
        timestamp = index * config.timestamp_step_seconds
    elif config.profile is LiveSoakProfile.STALE_CHURN:
        generation, position = divmod(
            index,
            config.flows,
        )

        generation_span = config.stale_after_seconds + config.flows * config.timestamp_step_seconds

        timestamp = generation * generation_span + position * config.timestamp_step_seconds
        flow_index = index
    else:
        flow_index = index % config.flows
        timestamp = index * config.timestamp_step_seconds

    host = (flow_index % 16_777_214) + 1

    source_address = f"10.{(host >> 16) & 255}.{(host >> 8) & 255}.{host & 255}"

    source_port = 1_024 + flow_index % 64_511

    return PacketMetadata(
        timestamp_seconds=timestamp,
        source_address=source_address,
        source_port=source_port,
        destination_address="192.0.2.1",
        destination_port=443,
        protocol=6,
        size=100,
    )


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run deterministic synthetic live-runtime sustained-load validation.")
    )

    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in LiveSoakProfile],
        default=LiveSoakProfile.STEADY.value,
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--flows",
        type=int,
        default=1_024,
    )
    parser.add_argument(
        "--max-tracked-flows",
        type=int,
        default=4_096,
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--timestamp-step",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the standalone synthetic soak command."""

    arguments = _build_parser().parse_args(argv)

    config = LiveSoakConfig(
        profile=LiveSoakProfile(arguments.profile),
        packets=arguments.packets,
        flows=arguments.flows,
        max_tracked_flows=(arguments.max_tracked_flows),
        stale_after_seconds=(arguments.stale_after),
        timestamp_step_seconds=(arguments.timestamp_step),
    )

    report = run_live_soak(config)

    write_live_soak_report(
        report,
        arguments.output,
    )

    print(
        json.dumps(
            report.as_dict(),
            indent=2,
            sort_keys=True,
        )
    )
