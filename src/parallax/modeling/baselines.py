"""Training-only majority and linear baselines for VNAT categories."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from parallax.data import FEATURE_COUNT, DatasetPartition, TrafficCategory
from parallax.modeling.dataset import FeaturePartition

BASELINE_EXPERIMENT_SCHEMA_VERSION: Final = "vnat-baseline-experiment-1"
CATEGORY_LABELS: Final = tuple(category.value for category in TrafficCategory)


class BaselineModel(StrEnum):
    """Initial reference models with intentionally limited complexity."""

    MAJORITY_CLASS = "majority-class"
    BALANCED_LOGISTIC_REGRESSION = "balanced-logistic-regression"


class BaselineModelingError(ValueError):
    """Raised when baseline fitting would violate the experiment contract."""


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Fixed, recorded settings for the initial linear baseline."""

    logistic_c: float = 1.0
    maximum_iterations: int = 2_000
    random_seed: int = 17

    def __post_init__(self) -> None:
        if not isfinite(self.logistic_c) or self.logistic_c <= 0.0:
            raise BaselineModelingError("logistic C must be finite and positive")
        if self.maximum_iterations < 1:
            raise BaselineModelingError("maximum iterations must be positive")
        if self.random_seed < 0:
            raise BaselineModelingError("random seed must not be negative")


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    """One category's validation classification metrics."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Deterministic closed-set metrics for one validation prediction vector."""

    accuracy: float
    balanced_accuracy: float
    micro_f1: float
    macro_f1: float
    categories: dict[str, CategoryMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible metric representation."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "categories": {
                category: asdict(metrics) for category, metrics in self.categories.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass(frozen=True, slots=True)
class TrainedBaseline:
    """One fitted estimator and its validation-only metrics."""

    model: BaselineModel
    estimator: Pipeline
    validation_metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """Fitted reference models without calibration or test-set access."""

    config: BaselineConfig
    training_windows: int
    validation_windows: int
    feature_count: int
    models: tuple[TrainedBaseline, ...]

    def model(self, model: BaselineModel) -> TrainedBaseline:
        """Return one named fitted baseline."""
        for selected in self.models:
            if selected.model is model:
                return selected
        raise KeyError(model)

    def as_dict(self) -> dict[str, object]:
        """Return deterministic experiment configuration and metrics."""
        return {
            "schema_version": BASELINE_EXPERIMENT_SCHEMA_VERSION,
            "configuration": {
                "logistic_c": self.config.logistic_c,
                "maximum_iterations": self.config.maximum_iterations,
                "random_seed": self.config.random_seed,
                "logistic_class_weight": "balanced",
                "logistic_solver": "lbfgs",
                "scaler_fit_partition": DatasetPartition.TRAIN.value,
            },
            "data": {
                "training_windows": self.training_windows,
                "validation_windows": self.validation_windows,
                "feature_count": self.feature_count,
                "evaluation_partition": DatasetPartition.VALIDATION.value,
            },
            "category_order": list(CATEGORY_LABELS),
            "models": {
                model.model.value: model.validation_metrics.as_dict() for model in self.models
            },
        }


def fit_initial_baselines(
    training: FeaturePartition,
    validation: FeaturePartition,
    *,
    config: BaselineConfig | None = None,
) -> BaselineExperimentResult:
    """Fit fixed baselines using training data and evaluate on validation only."""
    selected_config = config or BaselineConfig()
    _validate_partition(training, expected=DatasetPartition.TRAIN)
    _validate_partition(validation, expected=DatasetPartition.VALIDATION)
    if set(training.capture_ids).intersection(validation.capture_ids):
        raise BaselineModelingError("training and validation captures must not overlap")

    estimators = (
        (
            BaselineModel.MAJORITY_CLASS,
            Pipeline(
                steps=[
                    ("classifier", DummyClassifier(strategy="prior")),
                ]
            ),
        ),
        (
            BaselineModel.BALANCED_LOGISTIC_REGRESSION,
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=selected_config.logistic_c,
                            class_weight="balanced",
                            max_iter=selected_config.maximum_iterations,
                            random_state=selected_config.random_seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
    )

    fitted: list[TrainedBaseline] = []
    for model, estimator in estimators:
        estimator.fit(training.features, training.categories)
        predictions = np.asarray(estimator.predict(validation.features), dtype=np.str_)
        fitted.append(
            TrainedBaseline(
                model=model,
                estimator=estimator,
                validation_metrics=_classification_metrics(
                    validation.categories,
                    predictions,
                ),
            )
        )

    return BaselineExperimentResult(
        config=selected_config,
        training_windows=training.windows,
        validation_windows=validation.windows,
        feature_count=int(training.features.shape[1]),
        models=tuple(fitted),
    )


def _validate_partition(
    partition: FeaturePartition,
    *,
    expected: DatasetPartition,
) -> None:
    if partition.partition is not expected:
        raise BaselineModelingError(
            f"expected {expected.value} partition, got {partition.partition.value}"
        )
    if partition.features.ndim != 2 or partition.features.shape[0] < 1:
        raise BaselineModelingError("feature matrix must be nonempty and two-dimensional")
    if partition.features.shape[1] != FEATURE_COUNT:
        raise BaselineModelingError(f"feature matrix must contain {FEATURE_COUNT} columns")
    if partition.features.dtype != np.float32:
        raise BaselineModelingError("feature matrix must use float32 values")
    if not np.isfinite(partition.features).all():
        raise BaselineModelingError("feature matrix must contain only finite values")

    row_count = partition.features.shape[0]
    vectors = (
        partition.categories,
        partition.capture_ids,
        partition.vpn_statuses,
        partition.applications,
    )
    if any(vector.ndim != 1 or vector.shape[0] != row_count for vector in vectors):
        raise BaselineModelingError("partition provenance vectors must align with feature rows")
    if any(array.flags.writeable for array in (partition.features, *vectors)):
        raise BaselineModelingError("modeling partitions must be read-only")
    if set(partition.categories.tolist()) != set(CATEGORY_LABELS):
        raise BaselineModelingError("modeling partitions must contain every traffic category")
    if any(not capture_id for capture_id in partition.capture_ids):
        raise BaselineModelingError("capture identifiers must not be empty")


def _classification_metrics(
    expected: npt.NDArray[np.str_],
    predicted: npt.NDArray[np.str_],
) -> ClassificationMetrics:
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=CATEGORY_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(expected, predicted, labels=CATEGORY_LABELS)
    precision_values = cast("npt.NDArray[np.float64]", precision)
    recall_values = cast("npt.NDArray[np.float64]", recall)
    f1_values = cast("npt.NDArray[np.float64]", f1)
    support_values = cast("npt.NDArray[np.int64]", support)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(expected, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(expected, predicted)),
        micro_f1=float(f1_score(expected, predicted, labels=CATEGORY_LABELS, average="micro")),
        macro_f1=float(f1_score(expected, predicted, labels=CATEGORY_LABELS, average="macro")),
        categories={
            category: CategoryMetrics(
                precision=float(precision_values[index]),
                recall=float(recall_values[index]),
                f1=float(f1_values[index]),
                support=int(support_values[index]),
            )
            for index, category in enumerate(CATEGORY_LABELS)
        },
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
    )
