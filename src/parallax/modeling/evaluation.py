"""Pre-registered final evaluation for a frozen calibrated prototype model."""

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from parallax.data import DatasetPartition
from parallax.modeling.baselines import (
    CATEGORY_LABELS,
    BaselineModelingError,
    ClassificationMetrics,
    _classification_metrics,
    _validate_partition,
)
from parallax.modeling.bundle import LoadedPrototypeBundle
from parallax.modeling.calibration_bundle import LoadedPrototypeCalibration
from parallax.modeling.dataset import FeaturePartition
from parallax.modeling.episodes import CATEGORY_TO_INDEX
from parallax.modeling.inference import _expected_calibration_error
from parallax.modeling.scoring import (
    PrototypeScores,
    PrototypeScoringError,
    score_prototype_ood,
)

PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION: Final = "vnat-prototype-test-evaluation-1"
OOD_REVIEW_THRESHOLD: Final = 0.95
OOD_STRONG_THRESHOLD: Final = 0.99
OOD_THRESHOLDS: Final = (OOD_REVIEW_THRESHOLD, OOD_STRONG_THRESHOLD)


class PrototypeEvaluationError(ValueError):
    """Raised when final evaluation violates the frozen test protocol."""


@dataclass(frozen=True, slots=True)
class OODCategoryRate:
    """In-distribution flag rate for one known traffic category."""

    windows: int
    flagged_windows: int
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class OODThresholdResult:
    """Observed in-distribution false-positive behavior at one fixed threshold."""

    threshold: float
    flagged_windows: int
    false_positive_rate: float
    categories: dict[str, OODCategoryRate]

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible threshold result."""
        return {
            "threshold": self.threshold,
            "flagged_windows": self.flagged_windows,
            "false_positive_rate": self.false_positive_rate,
            "categories": {category: asdict(rate) for category, rate in self.categories.items()},
        }


@dataclass(frozen=True, slots=True)
class OODScoreSummary:
    """Fixed descriptive summary of OOD scores on known in-distribution data."""

    minimum: float
    quantile_05: float
    median: float
    quantile_95: float
    quantile_99: float
    maximum: float
    mean: float


@dataclass(frozen=True, slots=True)
class PrototypeTestEvaluation:
    """One-shot closed-set and ID false-positive test evidence."""

    classification_metrics: ClassificationMetrics
    expected_calibration_error: float
    ood_score_summary: OODScoreSummary
    ood_thresholds: tuple[OODThresholdResult, ...]
    test_windows: int
    test_captures: int

    def as_dict(self) -> dict[str, object]:
        """Return the complete pre-registered final evaluation payload."""
        return {
            "schema_version": PROTOTYPE_TEST_EVALUATION_SCHEMA_VERSION,
            "configuration": {
                "classification_probabilities": "raw-diagonal-mahalanobis-softmax",
                "probability_temperature_scaling": False,
                "expected_calibration_error_bins": 15,
                "ood_score": "one-minus-fitted-upper-tail-p-value",
                "ood_flag_comparison": "greater-than-or-equal",
                "ood_thresholds": list(OOD_THRESHOLDS),
            },
            "data": {
                "evaluation_partition": DatasetPartition.TEST.value,
                "test_windows": self.test_windows,
                "test_captures": self.test_captures,
                "known_traffic_only": True,
                "ood_examples_present": False,
                "model_selection_performed": False,
            },
            "category_order": list(CATEGORY_LABELS),
            "classification_metrics": self.classification_metrics.as_dict(),
            "expected_calibration_error": self.expected_calibration_error,
            "ood_score_summary": asdict(self.ood_score_summary),
            "ood_threshold_results": [result.as_dict() for result in self.ood_thresholds],
            "interpretation_boundary": {
                "classification_metrics": "closed-set performance on known VNAT categories",
                "ood_metrics": "known-traffic false-positive behavior only",
                "ood_detection_performance_measured": False,
            },
        }


def evaluate_prototype_test(
    bundle: LoadedPrototypeBundle,
    calibration: LoadedPrototypeCalibration,
    test: FeaturePartition,
) -> PrototypeTestEvaluation:
    """Run the frozen procedure once on the capture-isolated test partition."""
    try:
        _validate_partition(test, expected=DatasetPartition.TEST)
        if bundle.inference_config.expected_calibration_error_bins != 15:
            raise PrototypeEvaluationError("the frozen model must use 15 ECE bins")
        scores = score_prototype_ood(bundle, calibration, test.features)
    except (BaselineModelingError, PrototypeScoringError) as error:
        raise PrototypeEvaluationError(f"prototype test evaluation failed: {error}") from error

    expected_indices = np.asarray(
        [CATEGORY_TO_INDEX[category] for category in test.categories],
        dtype=np.int64,
    )
    evaluation = PrototypeTestEvaluation(
        classification_metrics=_classification_metrics(
            test.categories,
            scores.predicted_categories,
        ),
        expected_calibration_error=_expected_calibration_error(
            scores.class_probabilities,
            expected_indices,
            bins=bundle.inference_config.expected_calibration_error_bins,
        ),
        ood_score_summary=_score_summary(scores.ood_scores),
        ood_thresholds=tuple(
            _threshold_result(scores, test.categories, threshold=threshold)
            for threshold in OOD_THRESHOLDS
        ),
        test_windows=test.windows,
        test_captures=len(set(test.capture_ids.tolist())),
    )
    return evaluation


def _score_summary(scores: np.ndarray[tuple[int], np.dtype[np.float64]]) -> OODScoreSummary:
    quantiles = np.quantile(scores, [0.05, 0.5, 0.95, 0.99])
    return OODScoreSummary(
        minimum=float(scores.min()),
        quantile_05=float(quantiles[0]),
        median=float(quantiles[1]),
        quantile_95=float(quantiles[2]),
        quantile_99=float(quantiles[3]),
        maximum=float(scores.max()),
        mean=float(scores.mean()),
    )


def _threshold_result(
    scores: PrototypeScores,
    expected_categories: np.ndarray[tuple[int], np.dtype[np.str_]],
    *,
    threshold: float,
) -> OODThresholdResult:
    flagged = scores.ood_scores >= threshold
    return OODThresholdResult(
        threshold=threshold,
        flagged_windows=int(np.count_nonzero(flagged)),
        false_positive_rate=float(np.mean(flagged)),
        categories={
            category: _category_rate(
                flagged[expected_categories == category],
            )
            for category in CATEGORY_LABELS
        },
    )


def _category_rate(flagged: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> OODCategoryRate:
    return OODCategoryRate(
        windows=int(flagged.size),
        flagged_windows=int(np.count_nonzero(flagged)),
        false_positive_rate=float(np.mean(flagged)),
    )
