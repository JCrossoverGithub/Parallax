import numpy as np
import pandas as pd
import pytest

from parallax.data import (
    RAW_COLUMNS,
    BidirectionalFlowTracker,
    FlowPacketAssignment,
    IncrementalWindowTracker,
    ObservationWindow,
    PacketMetadata,
    RuntimeObservationWindow,
    RuntimeWindowError,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    extract_capture_windows,
    group_bidirectional_flows,
)

CAPTURE_NAME = "nonvpn_ssh_capture4.pcap"


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


def _batch_windows(
    packets: list[PacketMetadata],
    *,
    config: WindowExtractionConfig,
) -> list[ObservationWindow]:
    flows = group_bidirectional_flows(packets)
    frame = pd.DataFrame.from_records(
        [
            {
                "connection": flow.connection,
                "timestamps": list(flow.timestamps),
                "sizes": list(flow.sizes),
                "directions": list(flow.directions),
                "file_names": CAPTURE_NAME,
            }
            for flow in flows
        ],
        columns=RAW_COLUMNS,
    )
    return list(extract_capture_windows(frame, config=config))


def _runtime_windows(
    packets: list[PacketMetadata],
    *,
    config: WindowExtractionConfig,
) -> list[RuntimeObservationWindow]:
    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(CAPTURE_NAME, config=config)
    windows: list[RuntimeObservationWindow] = []

    for packet in packets:
        windows.extend(window_tracker.push(flow_tracker.push(packet)))

    windows.extend(window_tracker.finish())
    return windows


def _assert_window_equal(
    actual: RuntimeObservationWindow,
    expected: ObservationWindow,
) -> None:
    assert actual.window_id == expected.window_id
    assert actual.capture_id == expected.capture.capture_id
    assert actual.flow_id == expected.flow_id
    assert actual.connection == expected.connection
    assert actual.window_index == expected.window_index
    assert actual.start_offset_seconds == expected.start_offset_seconds
    assert actual.end_offset_seconds == expected.end_offset_seconds
    np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
    np.testing.assert_array_equal(actual.sizes, expected.sizes)
    np.testing.assert_array_equal(actual.directions, expected.directions)
    assert actual.timestamps.flags.writeable is False
    assert actual.sizes.flags.writeable is False
    assert actual.directions.flags.writeable is False


def test_incremental_windows_match_batch_window_contents() -> None:
    config = WindowExtractionConfig(
        window_seconds=10.0,
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    packets = [
        _packet(100.0, size=50),
        _packet(
            101.0,
            source="10.0.0.2",
            source_port=22,
            destination="10.0.0.1",
            destination_port=41_898,
            size=60,
        ),
        _packet(
            103.0,
            source="10.0.0.3",
            source_port=50_000,
            destination="10.0.0.4",
            destination_port=443,
            size=70,
        ),
        _packet(110.0, size=80),
        _packet(
            113.0,
            source="10.0.0.3",
            source_port=50_000,
            destination="10.0.0.4",
            destination_port=443,
            size=90,
        ),
        _packet(125.0, size=100),
    ]

    expected = {window.window_id: window for window in _batch_windows(packets, config=config)}
    actual = {window.window_id: window for window in _runtime_windows(packets, config=config)}

    assert actual.keys() == expected.keys()

    for window_id, expected_window in expected.items():
        _assert_window_equal(actual[window_id], expected_window)


def test_completed_window_emits_when_flow_enters_later_window() -> None:
    config = WindowExtractionConfig(
        window_seconds=10.0,
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(CAPTURE_NAME, config=config)

    assert window_tracker.push(flow_tracker.push(_packet(100.0))) == ()

    emitted = window_tracker.push(flow_tracker.push(_packet(110.0)))

    assert len(emitted) == 1
    assert emitted[0].window_index == 0
    np.testing.assert_array_equal(emitted[0].timestamps, [0.0])

    final = window_tracker.finish()

    assert len(final) == 1
    assert final[0].window_index == 1
    np.testing.assert_array_equal(final[0].timestamps, [0.0])


def test_exact_boundary_belongs_to_later_runtime_window() -> None:
    config = WindowExtractionConfig(
        window_seconds=40.96,
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    packets = [_packet(100.0), _packet(140.96)]

    windows = _runtime_windows(packets, config=config)

    assert [window.window_index for window in windows] == [0, 1]
    np.testing.assert_array_equal(windows[0].timestamps, [0.0])
    np.testing.assert_allclose(windows[1].timestamps, [0.0], atol=1e-12)


def test_release_threshold_excludes_twenty_and_retains_twenty_one() -> None:
    config = WindowExtractionConfig(
        window_seconds=10.0,
        minimum_packets=20,
        threshold_policy=WindowThresholdPolicy.RELEASE_COMPATIBLE,
    )

    twenty_packets = [_packet(100.0 + index * 0.1) for index in range(20)]
    twenty_packets.append(_packet(110.0))

    twenty_one_packets = [_packet(100.0 + index * 0.1) for index in range(21)]
    twenty_one_packets.append(_packet(110.0))

    twenty_windows = _runtime_windows(twenty_packets, config=config)
    twenty_one_windows = _runtime_windows(twenty_one_packets, config=config)

    assert [window.window_index for window in twenty_windows] == []
    assert [window.window_index for window in twenty_one_windows] == [0]


def test_finish_is_idempotent_and_closes_tracker() -> None:
    config = WindowExtractionConfig(
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(CAPTURE_NAME, config=config)

    window_tracker.push(flow_tracker.push(_packet(100.0)))

    assert len(window_tracker.finish()) == 1
    assert window_tracker.finish() == ()

    with pytest.raises(RuntimeWindowError, match="after runtime windowing is finished"):
        window_tracker.push(flow_tracker.push(_packet(101.0)))


def test_empty_tracker_finishes_without_windows() -> None:
    tracker = IncrementalWindowTracker(CAPTURE_NAME)

    assert tracker.finish() == ()
    assert tracker.finish() == ()


def test_rejects_empty_runtime_capture_id() -> None:
    with pytest.raises(
        RuntimeWindowError,
        match="runtime capture ID must not be empty",
    ):
        IncrementalWindowTracker("")


def test_rejects_nonsequential_flow_assignments() -> None:
    tracker = IncrementalWindowTracker(CAPTURE_NAME)
    packet = _packet(100.0)

    with pytest.raises(RuntimeWindowError, match="expected packet 1, got packet 2"):
        tracker.push(
            FlowPacketAssignment(
                packet_number=2,
                connection=packet.connection,
                direction=1,
                packet=packet,
            )
        )


def test_rejects_window_regression() -> None:
    config = WindowExtractionConfig(
        window_seconds=10.0,
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    tracker = IncrementalWindowTracker(CAPTURE_NAME, config=config)

    later = _packet(110.0)
    earlier = _packet(100.0)

    tracker.push(
        FlowPacketAssignment(
            packet_number=1,
            connection=later.connection,
            direction=1,
            packet=later,
        )
    )

    with pytest.raises(RuntimeWindowError, match="window index precedes"):
        tracker.push(
            FlowPacketAssignment(
                packet_number=2,
                connection=earlier.connection,
                direction=1,
                packet=earlier,
            )
        )


def test_global_window_advance_flushes_silent_flows() -> None:
    config = WindowExtractionConfig(
        window_seconds=10.0,
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )
    flow_tracker = BidirectionalFlowTracker()
    window_tracker = IncrementalWindowTracker(CAPTURE_NAME, config=config)

    first_flow = _packet(100.0)
    second_flow = _packet(
        101.0,
        source="10.0.0.3",
        source_port=50_000,
        destination="10.0.0.4",
        destination_port=443,
    )

    assert window_tracker.push(flow_tracker.push(first_flow)) == ()
    assert window_tracker.push(flow_tracker.push(second_flow)) == ()

    emitted = window_tracker.push(flow_tracker.push(_packet(110.0)))

    assert len(emitted) == 2
    assert {window.connection for window in emitted} == {
        first_flow.connection,
        second_flow.connection,
    }
    assert {window.window_index for window in emitted} == {0}
