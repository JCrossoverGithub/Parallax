from parallax.data import (
    IncrementalWindowTracker,
    PacketMetadata,
    RuntimeFlowTracker,
    RuntimeFlowTrackerConfig,
    WindowExtractionConfig,
)


def _packet(
    timestamp: float,
    source_port: int,
) -> PacketMetadata:
    return PacketMetadata(
        timestamp_seconds=timestamp,
        source_address="10.0.0.1",
        source_port=source_port,
        destination_address="10.0.0.2",
        destination_port=443,
        protocol=6,
        size=100,
    )


def test_flow_and_window_state_remain_bounded_under_sustained_churn() -> None:
    maximum_flows = 128
    generations = 32
    window_seconds = 10.0

    flow_tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            stale_after_seconds=window_seconds,
            max_tracked_flows=maximum_flows,
        )
    )
    window_tracker = IncrementalWindowTracker(
        "live:eth0:resource-bounds",
        config=WindowExtractionConfig(
            window_seconds=window_seconds,
            minimum_packets=1_000,
        ),
    )

    for generation in range(generations):
        generation_start = generation * 20.0

        for index in range(maximum_flows):
            unique_index = generation * maximum_flows + index

            assignment = flow_tracker.push(
                _packet(
                    generation_start + index * 0.001,
                    source_port=10_000 + unique_index,
                )
            )

            assert window_tracker.push(assignment) == ()

            assert flow_tracker.tracked_flow_count <= maximum_flows
            assert window_tracker.buffered_flow_count <= maximum_flows

    total_flows = maximum_flows * generations

    assert total_flows == 4_096

    assert flow_tracker.stats.packet_count == 4_096
    assert flow_tracker.stats.flows_created == 4_096
    assert flow_tracker.stats.flows_evicted_stale == 3_968
    assert flow_tracker.stats.capacity_rejections == 0
    assert flow_tracker.stats.peak_tracked_flow_count == 128

    assert flow_tracker.tracked_flow_count == 128
    assert window_tracker.buffered_flow_count == 128

    assert window_tracker.finish() == ()
    assert window_tracker.buffered_flow_count == 0
