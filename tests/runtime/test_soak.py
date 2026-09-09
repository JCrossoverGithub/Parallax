import json
from collections import deque
from collections.abc import Callable
from pathlib import Path

import pytest

from parallax.runtime.soak import (
    LIVE_SOAK_REPORT_SCHEMA_VERSION,
    LiveSoakConfig,
    LiveSoakError,
    LiveSoakProfile,
    main,
    run_live_soak,
    write_live_soak_report,
)


def _memory_probe(
    values: list[int],
) -> Callable[[], int]:
    remaining = deque(values)

    def probe() -> int:
        return remaining.popleft()

    return probe


def _wall_clock(
    values: list[float],
) -> Callable[[], float]:
    remaining = deque(values)

    def clock() -> float:
        return remaining.popleft()

    return clock


def test_default_config_matches_reference_profile() -> None:
    config = LiveSoakConfig()

    assert config.as_dict() == {
        "profile": "steady",
        "packets": 100_000,
        "flows": 1_024,
        "max_tracked_flows": 4_096,
        "stale_after_seconds": 120.0,
        "timestamp_step_seconds": 0.001,
        "synthetic_window_seconds": 10.0,
    }


def test_rejects_nonpositive_packet_count() -> None:
    with pytest.raises(
        LiveSoakError,
        match="packet count must be positive",
    ):
        LiveSoakConfig(packets=0)


def test_rejects_nonpositive_flow_count() -> None:
    with pytest.raises(
        LiveSoakError,
        match="flow count must be positive",
    ):
        LiveSoakConfig(flows=0)


def test_rejects_nonpositive_capacity() -> None:
    with pytest.raises(
        LiveSoakError,
        match="maximum tracked flows must be positive",
    ):
        LiveSoakConfig(max_tracked_flows=0)


@pytest.mark.parametrize(
    "stale_after_seconds",
    [0.0, float("nan")],
)
def test_rejects_invalid_stale_timeout(
    stale_after_seconds: float,
) -> None:
    with pytest.raises(
        LiveSoakError,
        match="stale flow timeout must be finite and positive",
    ):
        LiveSoakConfig(
            stale_after_seconds=stale_after_seconds,
        )


@pytest.mark.parametrize(
    "timestamp_step_seconds",
    [0.0, float("nan")],
)
def test_rejects_invalid_timestamp_step(
    timestamp_step_seconds: float,
) -> None:
    with pytest.raises(
        LiveSoakError,
        match="timestamp step must be finite and positive",
    ):
        LiveSoakConfig(
            timestamp_step_seconds=timestamp_step_seconds,
        )


def test_rejects_stale_timeout_shorter_than_soak_window() -> None:
    with pytest.raises(
        LiveSoakError,
        match="must be at least 10 seconds",
    ):
        LiveSoakConfig(
            stale_after_seconds=9.0,
        )


def test_steady_profile_must_fit_flow_capacity() -> None:
    with pytest.raises(
        LiveSoakError,
        match="flow count must not exceed",
    ):
        LiveSoakConfig(
            flows=5,
            max_tracked_flows=4,
        )


def test_steady_profile_must_refresh_before_stale_timeout() -> None:
    with pytest.raises(
        LiveSoakError,
        match="must revisit each flow",
    ):
        LiveSoakConfig(
            flows=10,
            stale_after_seconds=10.0,
            timestamp_step_seconds=1.0,
        )


def test_stale_churn_profile_must_fit_one_generation() -> None:
    with pytest.raises(
        LiveSoakError,
        match="flow count must not exceed",
    ):
        LiveSoakConfig(
            profile=LiveSoakProfile.STALE_CHURN,
            packets=10,
            flows=5,
            max_tracked_flows=4,
        )


def test_stale_churn_profile_requires_multiple_generations() -> None:
    with pytest.raises(
        LiveSoakError,
        match="requires more packets than flows",
    ):
        LiveSoakConfig(
            profile=LiveSoakProfile.STALE_CHURN,
            packets=4,
            flows=4,
            max_tracked_flows=4,
        )


def test_capacity_profile_requires_excess_distinct_flows() -> None:
    with pytest.raises(
        LiveSoakError,
        match="requires more flows than",
    ):
        LiveSoakConfig(
            profile=LiveSoakProfile.CAPACITY,
            packets=5,
            flows=4,
            max_tracked_flows=4,
        )


def test_capacity_profile_requires_enough_packets() -> None:
    with pytest.raises(
        LiveSoakError,
        match="requires enough packets",
    ):
        LiveSoakConfig(
            profile=LiveSoakProfile.CAPACITY,
            packets=4,
            flows=5,
            max_tracked_flows=4,
        )


def test_steady_soak_records_runtime_and_resource_stats() -> None:
    report = run_live_soak(
        LiveSoakConfig(
            packets=20,
            flows=2,
            max_tracked_flows=4,
        ),
        _memory_probe=_memory_probe([100, 125]),
        _wall_clock=_wall_clock([10.0, 12.0]),
    )

    assert report.outcome == "completed"
    assert report.source_packets_emitted == 20
    assert report.source_closed
    assert report.wall_elapsed_seconds == 2.0

    assert report.runtime_summary is not None
    assert report.runtime_summary.packets_processed == 20
    assert report.runtime_summary.events_emitted == 0

    assert report.pipeline_stats.packets_processed == 20
    assert report.pipeline_stats.tracked_flow_count == 2
    assert report.pipeline_stats.flows_created == 2
    assert report.pipeline_stats.flows_evicted_stale == 0
    assert report.pipeline_stats.capacity_rejections == 0
    assert report.pipeline_stats.peak_tracked_flow_count == 2
    assert report.pipeline_stats.prediction_events_emitted == 0
    assert report.pipeline_stats.finished

    payload = report.as_dict()

    assert payload["schema_version"] == (LIVE_SOAK_REPORT_SCHEMA_VERSION)

    memory = payload["memory"]

    assert isinstance(memory, dict)
    assert memory["peak_rss_growth_kib"] == 25


def test_stale_churn_soak_evicts_old_generations() -> None:
    report = run_live_soak(
        LiveSoakConfig(
            profile=LiveSoakProfile.STALE_CHURN,
            packets=6,
            flows=2,
            max_tracked_flows=2,
            stale_after_seconds=10.0,
            timestamp_step_seconds=0.1,
        ),
        _memory_probe=_memory_probe([200, 200]),
        _wall_clock=_wall_clock([1.0, 2.0]),
    )

    assert report.outcome == "completed"
    assert report.pipeline_stats.packets_processed == 6
    assert report.pipeline_stats.flows_created == 6
    assert report.pipeline_stats.flows_evicted_stale == 4
    assert report.pipeline_stats.capacity_rejections == 0
    assert report.pipeline_stats.tracked_flow_count == 2
    assert report.pipeline_stats.peak_tracked_flow_count == 2


def test_capacity_soak_records_expected_rejection() -> None:
    report = run_live_soak(
        LiveSoakConfig(
            profile=LiveSoakProfile.CAPACITY,
            packets=3,
            flows=3,
            max_tracked_flows=2,
        ),
        _memory_probe=_memory_probe([300, 310]),
        _wall_clock=_wall_clock([5.0, 6.0]),
    )

    assert report.outcome == "capacity_exceeded"
    assert report.source_packets_emitted == 3
    assert report.source_closed
    assert report.runtime_summary is None

    assert report.pipeline_stats.packets_processed == 2
    assert report.pipeline_stats.flows_created == 2
    assert report.pipeline_stats.capacity_rejections == 1
    assert report.pipeline_stats.peak_tracked_flow_count == 2
    assert not report.pipeline_stats.finished

    payload = report.as_dict()

    assert payload["runtime"] is None


def test_report_writer_is_atomic_json_output(
    tmp_path: Path,
) -> None:
    report = run_live_soak(
        LiveSoakConfig(
            packets=4,
            flows=1,
            max_tracked_flows=2,
        ),
        _memory_probe=_memory_probe([100, 100]),
        _wall_clock=_wall_clock([0.0, 1.0]),
    )

    output = tmp_path / "soak.json"

    assert (
        write_live_soak_report(
            report,
            output,
        )
        == output
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report.as_dict()


def test_report_writer_rejects_non_json_output(
    tmp_path: Path,
) -> None:
    report = run_live_soak(
        LiveSoakConfig(
            packets=2,
            flows=1,
            max_tracked_flows=2,
        ),
        _memory_probe=_memory_probe([100, 100]),
        _wall_clock=_wall_clock([0.0, 1.0]),
    )

    with pytest.raises(
        LiveSoakError,
        match=r"must use the \.json extension",
    ):
        write_live_soak_report(
            report,
            tmp_path / "soak.txt",
        )


def test_cli_writes_and_prints_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli-soak.json"

    main(
        [
            "--profile",
            "steady",
            "--packets",
            "4",
            "--flows",
            "1",
            "--max-tracked-flows",
            "2",
            "--output",
            str(output),
        ]
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))

    printed = json.loads(capsys.readouterr().out)

    assert persisted == printed
    assert persisted["outcome"] == "completed"


def test_capacity_profile_rejects_unexpected_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def complete_without_capacity(
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs

    monkeypatch.setattr(
        "parallax.runtime.soak.run_live_packet_predictions",
        complete_without_capacity,
    )

    with pytest.raises(
        LiveSoakError,
        match="completed without reaching flow capacity",
    ):
        run_live_soak(
            LiveSoakConfig(
                profile=LiveSoakProfile.CAPACITY,
                packets=3,
                flows=3,
                max_tracked_flows=2,
            ),
            _memory_probe=_memory_probe([100]),
            _wall_clock=_wall_clock([1.0]),
        )
