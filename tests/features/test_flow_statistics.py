import numpy as np
import numpy.typing as npt
import pytest

from parallax.features import (
    ACTIVITY_TIMEOUT_SECONDS,
    FeatureContractError,
    calculate_active_idle_feature_vector,
    calculate_interarrival_feature_vector,
)


def test_calculates_iat_statistics_in_release_schema_order() -> None:
    values = calculate_interarrival_feature_vector(
        timestamps=[0.0, 1.0, 3.0, 6.0],
        directions=[1, 0, 1, 0],
    )

    expected = np.asarray(
        [
            3.0,
            3.0,
            3.0,
            0.0,
            5.0,
            5.0,
            5.0,
            0.0,
            1.0,
            3.0,
            2.0,
            np.sqrt(2.0 / 3.0),
        ],
        dtype=np.float32,
    )
    assert np.array_equal(values, expected)
    assert values.dtype == np.float32
    assert values.flags.writeable is False


def test_direction_with_fewer_than_two_packets_uses_zero_statistics() -> None:
    values = calculate_interarrival_feature_vector(
        timestamps=[1.0, 2.0, 4.0],
        directions=[1, 0, 0],
    )

    assert np.array_equal(values[:4], np.zeros(4, dtype=np.float32))
    assert np.array_equal(values[4:8], np.asarray([2.0, 2.0, 2.0, 0.0]))


@pytest.mark.parametrize(
    ("timestamps", "directions", "message"),
    [
        ([[1.0, 2.0]], [[0, 1]], "one-dimensional"),
        ([1.0, 2.0], [[0, 1]], "one-dimensional"),
        ([], [], "cannot be empty"),
        ([1.0, 2.0], [1], "lengths do not match"),
        ([1.0, float("nan")], [0, 1], "must all be finite"),
        ([2.0, 1.0], [0, 1], "must be nondecreasing"),
        ([1.0, 2.0], [0.0, 1.0], "must be integers"),
        ([1.0, 2.0], [0, 2], "must be zero or one"),
    ],
)
def test_rejects_invalid_packet_timing(
    timestamps: npt.ArrayLike,
    directions: npt.ArrayLike,
    message: str,
) -> None:
    with pytest.raises(FeatureContractError, match=message):
        calculate_interarrival_feature_vector(timestamps, directions)


def test_rejects_nonnumeric_timestamps() -> None:
    with pytest.raises(FeatureContractError, match="timestamps must be numeric"):
        calculate_interarrival_feature_vector(["invalid"], [0])


def test_calculates_active_and_excess_idle_statistics() -> None:
    values = calculate_active_idle_feature_vector(
        timestamps=[0.0, 1.0, 7.0, 8.0, 20.0],
    )

    assert ACTIVITY_TIMEOUT_SECONDS == 5.0
    assert np.array_equal(
        values,
        np.asarray(
            [
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                7.0,
                4.0,
                3.0,
            ],
            dtype=np.float32,
        ),
    )
    assert values.dtype == np.float32
    assert values.flags.writeable is False


def test_unsplit_flow_has_one_active_period_and_zero_idle_statistics() -> None:
    values = calculate_active_idle_feature_vector([1.0, 2.0, 4.0])

    assert np.array_equal(values[:4], np.asarray([3.0, 3.0, 3.0, 0.0]))
    assert np.array_equal(values[4:], np.zeros(4, dtype=np.float32))


def test_gap_equal_to_timeout_remains_active() -> None:
    values = calculate_active_idle_feature_vector([0.0, 5.0])

    assert np.array_equal(values[:4], np.asarray([5.0, 5.0, 5.0, 0.0]))
    assert np.array_equal(values[4:], np.zeros(4, dtype=np.float32))


def test_zero_length_active_periods_are_omitted() -> None:
    values = calculate_active_idle_feature_vector([0.0, 6.0, 12.0])

    assert np.array_equal(values[:4], np.zeros(4, dtype=np.float32))
    assert np.array_equal(values[4:], np.asarray([1.0, 1.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ([[1.0, 2.0]], "one-dimensional"),
        ([], "cannot be empty"),
        ([1.0, float("inf")], "must all be finite"),
        ([2.0, 1.0], "must be nondecreasing"),
    ],
)
def test_active_idle_rejects_invalid_timestamps(
    timestamps: npt.ArrayLike,
    message: str,
) -> None:
    with pytest.raises(FeatureContractError, match=message):
        calculate_active_idle_feature_vector(timestamps)
