from math import inf, nan

import pytest

from parallax.data import (
    BidirectionalFlowTracker,
    FlowConstructionError,
    PacketMetadata,
    RuntimeFlowTracker,
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
