import numpy as np
import numpy.typing as npt
import pytest

from parallax.features import (
    FeatureContractError,
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
