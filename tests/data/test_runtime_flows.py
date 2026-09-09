from math import inf, nan

import pytest

from parallax.data import (
    BidirectionalFlowTracker,
    FlowConstructionError,
    PacketMetadata,
    RuntimeFlowCapacityError,
    RuntimeFlowTracker,
    RuntimeFlowTrackerConfig,
    RuntimeFlowTrackerStats,
)


def _packet(
    timestamp: float,
    *,
    source: str = "10.0.0.1",
    source_port: int = 50_000,
    destination: str = "10.0.0.2",
    destination_port: int = 443,
    protocol: int = 6,
    size: int = 100,
) -> PacketMetadata:
    return PacketMetadata(
        timestamp_seconds=timestamp,
        source_address=source,
        source_port=source_port,
        destination_address=destination,
        destination_port=destination_port,
        protocol=protocol,
        size=size,
    )


def test_runtime_tracker_matches_batch_flow_orientation() -> None:
    packets = [
        _packet(1.0),
        _packet(
            2.0,
            source="10.0.0.2",
            source_port=443,
            destination="10.0.0.1",
            destination_port=50_000,
        ),
        _packet(3.0),
        _packet(
            4.0,
            protocol=17,
        ),
    ]

    batch = BidirectionalFlowTracker()
    runtime = RuntimeFlowTracker()

    for packet in packets:
        expected = batch.push(packet)
        actual = runtime.push(packet)

        assert actual.packet_number == expected.packet_number
        assert actual.connection == expected.connection
        assert actual.direction == expected.direction
        assert actual.packet is packet

    assert runtime.packet_count == 4
    assert runtime.tracked_flow_count == 2


def test_runtime_tracker_does_not_grow_flow_count_for_one_flow() -> None:
    tracker = RuntimeFlowTracker()

    for index in range(1_000):
        tracker.push(
            _packet(
                float(index),
                source=("10.0.0.1" if index % 2 == 0 else "10.0.0.2"),
                source_port=(50_000 if index % 2 == 0 else 443),
                destination=("10.0.0.2" if index % 2 == 0 else "10.0.0.1"),
                destination_port=(443 if index % 2 == 0 else 50_000),
            )
        )

    assert tracker.packet_count == 1_000
    assert tracker.tracked_flow_count == 1


@pytest.mark.parametrize("timestamp", [nan, inf, -inf])
def test_runtime_tracker_rejects_nonfinite_timestamp(
    timestamp: float,
) -> None:
    tracker = RuntimeFlowTracker()

    with pytest.raises(
        FlowConstructionError,
        match="timestamp must be finite",
    ):
        tracker.push(_packet(timestamp))

    assert tracker.packet_count == 0
    assert tracker.tracked_flow_count == 0


def test_runtime_tracker_rejects_decreasing_timestamp() -> None:
    tracker = RuntimeFlowTracker()

    tracker.push(_packet(2.0))

    with pytest.raises(
        FlowConstructionError,
        match="timestamp precedes the previous packet",
    ):
        tracker.push(_packet(1.0))

    assert tracker.packet_count == 1
    assert tracker.tracked_flow_count == 1


@pytest.mark.parametrize("size", [0, -1])
def test_runtime_tracker_rejects_nonpositive_size(size: int) -> None:
    tracker = RuntimeFlowTracker()

    with pytest.raises(
        FlowConstructionError,
        match="size must be greater than zero",
    ):
        tracker.push(_packet(1.0, size=size))

    assert tracker.packet_count == 0
    assert tracker.tracked_flow_count == 0


@pytest.mark.parametrize(
    "stale_after_seconds",
    [0.0, -1.0, inf, -inf, nan],
)
def test_rejects_invalid_stale_timeout(
    stale_after_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="stale flow timeout must be finite and positive",
    ):
        RuntimeFlowTrackerConfig(
            stale_after_seconds=stale_after_seconds,
        )


@pytest.mark.parametrize("max_tracked_flows", [0, -1])
def test_rejects_invalid_maximum_flow_capacity(
    max_tracked_flows: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="maximum tracked flows must be positive",
    ):
        RuntimeFlowTrackerConfig(
            max_tracked_flows=max_tracked_flows,
        )


def test_stale_flows_are_evicted_before_new_assignment() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            stale_after_seconds=10.0,
        )
    )

    tracker.push(
        _packet(
            0.0,
            source_port=10_001,
        )
    )
    tracker.push(
        _packet(
            5.0,
            source_port=10_002,
        )
    )

    assert tracker.tracked_flow_count == 2

    tracker.push(
        _packet(
            10.0,
            source_port=10_003,
        )
    )

    assert tracker.tracked_flow_count == 2
    assert tracker.stats.flows_created == 3
    assert tracker.stats.flows_evicted_stale == 1
    assert tracker.stats.peak_tracked_flow_count == 2


def test_existing_flow_refreshes_stale_deadline() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            stale_after_seconds=10.0,
        )
    )

    tracker.push(_packet(0.0))
    tracker.push(_packet(9.0))

    tracker.push(
        _packet(
            10.0,
            source_port=50_001,
        )
    )

    assert tracker.tracked_flow_count == 2
    assert tracker.stats.flows_evicted_stale == 0


def test_packet_after_stale_boundary_starts_new_flow_orientation() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            stale_after_seconds=10.0,
        )
    )

    first = tracker.push(_packet(0.0))

    second = tracker.push(
        _packet(
            10.0,
            source="10.0.0.2",
            source_port=443,
            destination="10.0.0.1",
            destination_port=50_000,
        )
    )

    assert first.direction == 1
    assert second.direction == 1
    assert second.connection != first.connection

    assert tracker.stats.flows_created == 2
    assert tracker.stats.flows_evicted_stale == 1
    assert tracker.tracked_flow_count == 1


def test_capacity_rejects_new_flow_without_advancing_packet_state() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            max_tracked_flows=1,
        )
    )

    tracker.push(_packet(1.0))

    with pytest.raises(
        RuntimeFlowCapacityError,
        match="runtime flow capacity reached: 1",
    ):
        tracker.push(
            _packet(
                2.0,
                source_port=50_001,
            )
        )

    assert tracker.stats == RuntimeFlowTrackerStats(
        packet_count=1,
        tracked_flow_count=1,
        flows_created=1,
        flows_evicted_stale=0,
        capacity_rejections=1,
        peak_tracked_flow_count=1,
    )


def test_capacity_still_allows_packets_for_existing_flow() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            max_tracked_flows=1,
        )
    )

    tracker.push(_packet(1.0))

    assignment = tracker.push(
        _packet(
            2.0,
            source="10.0.0.2",
            source_port=443,
            destination="10.0.0.1",
            destination_port=50_000,
        )
    )

    assert assignment.direction == 0
    assert tracker.packet_count == 2
    assert tracker.stats.capacity_rejections == 0


def test_stale_eviction_frees_capacity_before_new_flow() -> None:
    tracker = RuntimeFlowTracker(
        config=RuntimeFlowTrackerConfig(
            stale_after_seconds=10.0,
            max_tracked_flows=1,
        )
    )

    tracker.push(_packet(0.0))

    assignment = tracker.push(
        _packet(
            10.0,
            source_port=50_001,
        )
    )

    assert assignment.packet_number == 2
    assert tracker.stats == RuntimeFlowTrackerStats(
        packet_count=2,
        tracked_flow_count=1,
        flows_created=2,
        flows_evicted_stale=1,
        capacity_rejections=0,
        peak_tracked_flow_count=1,
    )
