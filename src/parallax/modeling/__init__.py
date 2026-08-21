"""Leakage-resistant modeling contracts for Parallax."""

from parallax.modeling.baselines import (
    BASELINE_EXPERIMENT_SCHEMA_VERSION,
    CATEGORY_LABELS,
    BaselineConfig,
    BaselineExperimentResult,
    BaselineModel,
    BaselineModelingError,
    CategoryMetrics,
    ClassificationMetrics,
    TrainedBaseline,
    fit_initial_baselines,
)
from parallax.modeling.dataset import (
    MODELING_DATASET_SCHEMA_VERSION,
    FeaturePartition,
    ModelingDatasetError,
    PartitionedFeatureDataset,
    load_partitioned_feature_dataset,
)
from parallax.modeling.report import (
    BASELINE_VALIDATION_REPORT_SCHEMA_VERSION,
    BaselineValidationReport,
    BaselineValidationReportError,
    export_baseline_validation_report,
)

__all__ = [
    "BASELINE_EXPERIMENT_SCHEMA_VERSION",
    "BASELINE_VALIDATION_REPORT_SCHEMA_VERSION",
    "CATEGORY_LABELS",
    "MODELING_DATASET_SCHEMA_VERSION",
    "BaselineConfig",
    "BaselineExperimentResult",
    "BaselineModel",
    "BaselineModelingError",
    "BaselineValidationReport",
    "BaselineValidationReportError",
    "CategoryMetrics",
    "ClassificationMetrics",
    "FeaturePartition",
    "ModelingDatasetError",
    "PartitionedFeatureDataset",
    "TrainedBaseline",
    "export_baseline_validation_report",
    "fit_initial_baselines",
    "load_partitioned_feature_dataset",
]
