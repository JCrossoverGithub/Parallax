from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from parallax.data.windowing import (
    ObservationWindow,
    VnatWindowError,
    WindowExtractionConfig,
    WindowThresholdPolicy,
    extract_capture_windows,
)

RAW_COLUMNS = ("connection", "timestamps", "sizes", "directions", "file_names")
CONNECTION: tuple[str, int, str, int, int] = ("10.0.0.1", 1234, "10.0.0.2", 443, 6)
CAPTURE = "nonvpn_youtube_capture1.pcap"


def make_frame(
    *,
    connection: object = CONNECTION,
    timestamps: object | None = None,
    sizes: object | None = None,
    directions: object | None = None,
    file_name: object = CAPTURE,
) -> pd.DataFrame:
    selected_timestamps = list(range(21)) if timestamps is None else timestamps
    count = len(selected_timestamps) if isinstance(selected_timestamps, Sequence) else 21
    selected_sizes = list(range(100, 100 + count)) if sizes is None else sizes
    selected_directions = (
        [index % 2 for index in range(count)] if directions is None else directions
    )
    return pd.DataFrame(
        [[connection, selected_timestamps, selected_sizes, selected_directions, file_name]],
        columns=RAW_COLUMNS,
    )


def test_extracts_capture_aligned_connection_windows_in_stable_time_order() -> None:
    first_times = [index * 0.01 for index in range(21)]
    second_times = [40.96 + index * 0.01 for index in range(21)]
    timestamps = list(reversed(first_times + second_times))
    sizes = list(range(len(timestamps)))
    directions = [value % 2 for value in sizes]
    expected_pairs = sorted(zip(timestamps, sizes, directions, strict=True))
    frame = make_frame(timestamps=timestamps, sizes=sizes, directions=directions)

    windows = list(extract_capture_windows(frame))

    assert len(windows) == 2
    assert [window.window_index for window in windows] == [0, 1]
    assert [window.packet_count for window in windows] == [21, 21]
    assert [window.start_offset_seconds for window in windows] == [0.0, 40.96]
    assert [window.end_offset_seconds for window in windows] == [40.96, 81.92]
    assert windows[0].capture.capture_id == CAPTURE
    assert windows[0].connection == CONNECTION
    assert windows[0].flow_id == windows[1].flow_id
    assert windows[0].window_id.endswith(":0")
    assert windows[1].window_id.endswith(":1")
    assert windows[0].timestamps.flags.writeable is False
    assert windows[0].sizes.flags.writeable is False
    assert windows[0].directions.flags.writeable is False

    ordered_sizes = [pair[1] for pair in expected_pairs]
    ordered_directions = [pair[2] for pair in expected_pairs]
    np.testing.assert_array_equal(windows[0].sizes, ordered_sizes[:21])
    np.testing.assert_array_equal(windows[1].sizes, ordered_sizes[21:])
    np.testing.assert_array_equal(windows[0].directions, ordered_directions[:21])
    np.testing.assert_array_equal(windows[1].directions, ordered_directions[21:])
    np.testing.assert_allclose(windows[0].timestamps, np.arange(21) * 0.01)
    np.testing.assert_allclose(windows[1].timestamps, np.arange(21) * 0.01)


def test_stable_sort_preserves_packet_order_for_equal_timestamps() -> None:
    timestamps = [100.0] * 21
    sizes = list(range(21))
    frame = make_frame(timestamps=timestamps, sizes=sizes)

    window = next(extract_capture_windows(frame))

    np.testing.assert_array_equal(window.sizes, sizes)


def test_release_threshold_excludes_exactly_twenty_packets() -> None:
    frame = make_frame(timestamps=list(range(20)))

    assert list(extract_capture_windows(frame)) == []


def test_paper_literal_threshold_includes_exactly_twenty_packets() -> None:
    frame = make_frame(timestamps=list(range(20)))
    config = WindowExtractionConfig(threshold_policy=WindowThresholdPolicy.PAPER_LITERAL)

    windows = list(extract_capture_windows(frame, config=config))

    assert len(windows) == 1
    assert windows[0].packet_count == 20


def test_capture_origin_is_shared_across_connections() -> None:
    first = make_frame(timestamps=[100.0 + index * 0.01 for index in range(21)])
    second = make_frame(
        connection=("10.0.0.3", 2345, "10.0.0.4", 22, 6),
        timestamps=[141.0 + index * 0.01 for index in range(21)],
    )
    frame = pd.concat([second, first], ignore_index=True)

    windows = list(extract_capture_windows(frame))

    assert sorted(window.window_index for window in windows) == [0, 1]
    assert len({window.flow_id for window in windows}) == 2
    assert len({window.window_id for window in windows}) == 2


def test_identifiers_are_deterministic() -> None:
    frame = make_frame()

    first = next(extract_capture_windows(frame))
    second = next(extract_capture_windows(frame.copy()))

    assert first.flow_id == second.flow_id
    assert first.window_id == second.window_id


def test_output_order_is_independent_of_input_row_order() -> None:
    first = make_frame()
    second = make_frame(
        connection=("10.0.0.3", 2345, "10.0.0.4", 22, 6),
    )
    forward = pd.concat([first, second], ignore_index=True)
    reverse = pd.concat([second, first], ignore_index=True)

    forward_ids = [window.window_id for window in extract_capture_windows(forward)]
    reverse_ids = [window.window_id for window in extract_capture_windows(reverse)]

    assert forward_ids == reverse_ids


def test_exact_window_boundary_belongs_to_the_later_window() -> None:
    frame = make_frame(
        timestamps=[0.0, 40.96],
        sizes=[100, 200],
        directions=[0, 1],
    )
    config = WindowExtractionConfig(
        minimum_packets=1,
        threshold_policy=WindowThresholdPolicy.PAPER_LITERAL,
    )

    windows = list(extract_capture_windows(frame, config=config))

    assert [window.window_index for window in windows] == [0, 1]
    np.testing.assert_array_equal(windows[0].timestamps, [0.0])
    np.testing.assert_array_equal(windows[1].timestamps, [0.0])


def test_large_epoch_timestamps_remain_inside_their_window() -> None:
    origin = 1_563_289_706.330096
    timestamps = [origin + index * 0.01 for index in range(21)]
    timestamps.extend(origin + 40.97 + index * 0.01 for index in range(21))
    frame = make_frame(timestamps=timestamps)

    windows = list(extract_capture_windows(frame))

    assert len(windows) == 2
    for window in windows:
        assert np.all(window.timestamps >= 0.0)
        assert np.all(window.timestamps < 40.96)


@pytest.mark.parametrize("window_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_config_rejects_invalid_window_seconds(window_seconds: float) -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        WindowExtractionConfig(window_seconds=window_seconds)


def test_config_rejects_invalid_minimum_packets() -> None:
    with pytest.raises(ValueError, match="minimum_packets"):
        WindowExtractionConfig(minimum_packets=0)


def test_config_rejects_invalid_threshold_policy() -> None:
    with pytest.raises(TypeError, match="threshold_policy"):
        WindowExtractionConfig(threshold_policy="release-compatible")  # type: ignore[arg-type]


def test_rejects_wrong_columns() -> None:
    frame = make_frame().drop(columns="directions")

    with pytest.raises(VnatWindowError, match="expected raw columns"):
        extract_capture_windows(frame)


def test_rejects_empty_frame() -> None:
    frame = pd.DataFrame(columns=RAW_COLUMNS)

    with pytest.raises(VnatWindowError, match="empty capture"):
        extract_capture_windows(frame)


@pytest.mark.parametrize("file_name", [123, None])
def test_rejects_non_string_capture_name(file_name: object) -> None:
    frame = make_frame(file_name=file_name)

    with pytest.raises(VnatWindowError, match="exactly one capture filename"):
        extract_capture_windows(frame)


def test_rejects_multiple_capture_names() -> None:
    frame = pd.concat(
        [make_frame(), make_frame(file_name="vpn_youtube_capture2.pcap")],
        ignore_index=True,
    )

    with pytest.raises(VnatWindowError, match="exactly one capture filename"):
        extract_capture_windows(frame)


def test_rejects_duplicate_connections() -> None:
    frame = pd.concat([make_frame(), make_frame()], ignore_index=True)

    with pytest.raises(VnatWindowError, match="duplicate connection"):
        extract_capture_windows(frame)


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        (("10.0.0.1", 1), "five-tuple"),
        ((1, 1234, "10.0.0.2", 443, 6), "addresses must be strings"),
        (("10.0.0.1", True, "10.0.0.2", 443, 6), "must be integers"),
        (("10.0.0.1", "1234", "10.0.0.2", 443, 6), "must be integers"),
        (("10.0.0.1", -1, "10.0.0.2", 443, 6), "port is out of range"),
        (("10.0.0.1", 1, "10.0.0.2", 65_536, 6), "port is out of range"),
        (("10.0.0.1", 1, "10.0.0.2", 443, 256), "protocol is out of range"),
    ],
)
def test_rejects_invalid_connection(connection: object, message: str) -> None:
    frame = make_frame(connection=connection)

    with pytest.raises(VnatWindowError, match=message):
        extract_capture_windows(frame)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamps", ["invalid"] * 21, "timestamps must be numeric"),
        ("timestamps", [[1.0] * 21], "one-dimensional"),
        ("timestamps", [], "cannot be empty"),
        ("timestamps", [0.0] * 20 + [float("inf")], "timestamps must be finite"),
        ("sizes", [1.5] * 21, "packet sizes must be integers"),
        ("sizes", [1] * 20 + [-1], "packet sizes cannot be negative"),
        ("directions", [0.5] * 21, "packet directions must be integers"),
        ("directions", [0] * 20 + [2], "must be zero or one"),
        ("directions", [0] * 20, "lengths do not match"),
    ],
)
def test_rejects_invalid_packet_arrays(field: str, value: object, message: str) -> None:
    frame = make_frame()
    frame.at[0, field] = cast(Any, value)

    with pytest.raises(VnatWindowError, match=message):
        extract_capture_windows(frame)


def test_observation_window_has_identity_equality() -> None:
    window = next(extract_capture_windows(make_frame()))

    assert isinstance(window, ObservationWindow)
    assert window != next(extract_capture_windows(make_frame()))
