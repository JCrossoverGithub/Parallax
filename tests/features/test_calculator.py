import numpy as np
import numpy.typing as npt
import pytest

from parallax.features import (
    ByteTotalPolicy,
    FeatureCalculationConfig,
    FeatureContractError,
    build_directional_size_signals,
    calculate_active_idle_feature_vector,
    calculate_aggregate_feature_vector,
    calculate_feature_vector,
    calculate_interarrival_feature_vector,
    calculate_wavelet_feature_vector,
)


def test_builds_directional_size_signals_with_bin_aggregation() -> None:
    incoming, outgoing = build_directional_size_signals(
        timestamps=[0.001, 0.009, 0.010, 40.959],
        sizes=[10, 20, 30, 40],
        directions=[0, 0, 1, 1],
    )

    assert incoming.shape == (4096,)
    assert outgoing.shape == (4096,)
    assert incoming[0] == 30.0
    assert outgoing[1] == 30.0
    assert outgoing[4095] == 40.0
    assert incoming.sum() == 30.0
    assert outgoing.sum() == 70.0
    assert incoming.flags.writeable is False
    assert outgoing.flags.writeable is False


def test_calculates_complete_vector_in_official_schema_order() -> None:
    timestamps = np.asarray([0.001, 0.009, 0.010, 6.0, 6.01])
    sizes = np.asarray([10, 20, 30, 40, 50])
    directions = np.asarray([0, 0, 1, 1, 0])
    config = FeatureCalculationConfig()
    incoming, outgoing = build_directional_size_signals(
        timestamps,
        sizes,
        directions,
        config=config,
    )
    expected = np.concatenate(
        (
            calculate_interarrival_feature_vector(timestamps, directions),
            calculate_active_idle_feature_vector(timestamps),
            calculate_aggregate_feature_vector(
                sizes,
                directions,
                window_seconds=config.window_seconds,
                byte_total_policy=config.byte_total_policy,
            ),
            calculate_wavelet_feature_vector(incoming, outgoing),
        )
    ).astype(np.float32)

    values = calculate_feature_vector(
        timestamps,
        sizes,
        directions,
        config=config,
    )

    assert values.shape == (129,)
    assert np.array_equal(values, expected)
    assert values.flags.writeable is False


def test_corrected_config_changes_only_duplicated_byte_fields() -> None:
    timestamps = [0.0, 1.0, 2.0]
    sizes = [100, 200, 300]
    directions = [1, 0, 1]
    release = calculate_feature_vector(timestamps, sizes, directions)
    corrected = calculate_feature_vector(
        timestamps,
        sizes,
        directions,
        config=FeatureCalculationConfig(
            byte_total_policy=ByteTotalPolicy.CORRECTED,
        ),
    )

    differing = np.flatnonzero(release != corrected)
    assert np.array_equal(differing, np.asarray([23, 24]))


@pytest.mark.parametrize(
    ("window_seconds", "time_bin_seconds", "message"),
    [
        (0.0, 0.01, "window_seconds"),
        (float("inf"), 0.01, "window_seconds"),
        (40.96, 0.0, "time_bin_seconds"),
        (40.96, float("nan"), "time_bin_seconds"),
        (10.0, 0.01, "must produce 4096 samples"),
    ],
)
def test_rejects_invalid_calculation_config(
    window_seconds: float,
    time_bin_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FeatureCalculationConfig(
            window_seconds=window_seconds,
            time_bin_seconds=time_bin_seconds,
        )


def test_rejects_invalid_config_policy() -> None:
    with pytest.raises(TypeError, match="must be a ByteTotalPolicy"):
        FeatureCalculationConfig(
            byte_total_policy="corrected",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("timestamps", "sizes", "directions", "message"),
    [
        ([[0.0]], [[100]], [[1]], "one-dimensional"),
        ([], [], [], "cannot be empty"),
        ([0.0, 1.0], [100], [1, 0], "lengths do not match"),
        ([0.0, float("nan")], [100, 200], [1, 0], "must all be finite"),
        ([1.0, 0.0], [100, 200], [1, 0], "must be nondecreasing"),
        ([-0.1], [100], [1], "within the observation window"),
        ([40.96], [100], [1], "within the observation window"),
        ([0.0], [100.0], [1], "sizes must be integers"),
        ([0.0], [100], [1.0], "directions must be integers"),
        ([0.0], [-1], [1], "sizes cannot be negative"),
        ([0.0], [100], [2], "directions must be zero or one"),
    ],
)
def test_signal_builder_rejects_invalid_window_packets(
    timestamps: npt.ArrayLike,
    sizes: npt.ArrayLike,
    directions: npt.ArrayLike,
    message: str,
) -> None:
    with pytest.raises(FeatureContractError, match=message):
        build_directional_size_signals(timestamps, sizes, directions)


def test_signal_builder_rejects_nonnumeric_timestamps() -> None:
    with pytest.raises(FeatureContractError, match="timestamps must be numeric"):
        build_directional_size_signals(["invalid"], [100], [1])
