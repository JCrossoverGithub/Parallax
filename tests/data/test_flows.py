from math import inf, nan

import pytest

from parallax.data import (
    BidirectionalFlow,
    FlowConstructionError,
    PacketMetadata,
    group_bidirectional_flows,
)


def _packet(
    timestamp: float,
    *,
    source: str = "10.0.0.1",
    source_port: int = 41_898,
    destination: str = "10.0.0.2",
    destination_port: int = 22,
    protocol: int = 6,
    size: int = 52,
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


def test_groups_both_directions_under_the_first_observed_connection() -> None:
    packets = [
        _packet(1.0, size=52),
        _packet(
            2.0,
            source="10.0.0.2",
            source_port=22,
            destination="10.0.0.1",
            destination_port=41_898,
            size=1_378,
        ),
        _packet(3.0, size=60),
    ]

    flows = group_bidirectional_flows(packets)

    assert flows == (
        BidirectionalFlow(
            connection=("10.0.0.1", 41_898, "10.0.0.2", 22, 6),
            timestamps=(1.0, 2.0, 3.0),
            sizes=(52, 1_378, 60),
            directions=(1, 0, 1),
        ),
    )
    assert flows[0].packet_count == 3


def test_reverse_packet_can_define_the_forward_orientation() -> None:
    packets = [
        _packet(
            1.0,
            source="10.0.0.2",
            source_port=22,
            destination="10.0.0.1",
            destination_port=41_898,
        ),
        _packet(2.0),
    ]

    flow = group_bidirectional_flows(packets)[0]

    assert flow.connection == ("10.0.0.2", 22, "10.0.0.1", 41_898, 6)
    assert flow.directions == (1, 0)


def test_keeps_protocols_separate_and_orders_flows_by_first_observation() -> None:
    packets = [
        _packet(1.0, protocol=17, size=32),
        _packet(1.0, protocol=6),
        _packet(2.0, protocol=17, size=40),
    ]

    flows = group_bidirectional_flows(packets)

    assert [flow.connection[-1] for flow in flows] == [17, 6]
    assert [flow.packet_count for flow in flows] == [2, 1]
    assert flows[0].timestamps == (1.0, 2.0)
    assert flows[1].timestamps == (1.0,)


def test_empty_packet_stream_produces_no_flows() -> None:
    assert group_bidirectional_flows([]) == ()


@pytest.mark.parametrize("timestamp", [nan, inf, -inf])
def test_rejects_nonfinite_timestamps(timestamp: float) -> None:
    with pytest.raises(FlowConstructionError, match="packet 1: timestamp must be finite"):
        group_bidirectional_flows([_packet(timestamp)])


def test_rejects_decreasing_timestamps() -> None:
    with pytest.raises(FlowConstructionError, match="packet 2: timestamp precedes"):
        group_bidirectional_flows([_packet(2.0), _packet(1.0)])


@pytest.mark.parametrize("size", [0, -1])
def test_rejects_nonpositive_packet_sizes(size: int) -> None:
    with pytest.raises(FlowConstructionError, match="packet 1: size must be greater than zero"):
        group_bidirectional_flows([_packet(1.0, size=size)])
