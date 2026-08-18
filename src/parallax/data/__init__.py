"""Dataset contracts and ingestion helpers for Parallax."""

from parallax.data.inspection import (
    VnatDatasetError,
    VnatDatasetSummary,
    VnatInspectionReport,
    inspect_raw_dataframe,
    inspect_vnat_file,
)
from parallax.data.vnat import (
    APPLICATION_TO_CATEGORY,
    FEATURE_COUNT,
    FEATURE_LABEL_COLUMN,
    MIN_PACKETS_PER_WINDOW,
    RAW_COLUMNS,
    TIME_BIN_SECONDS,
    WINDOW_SECONDS,
    Application,
    CaptureMetadata,
    TrafficCategory,
    VnatFilenameError,
    VpnStatus,
    parse_capture_filename,
)

__all__ = [
    "APPLICATION_TO_CATEGORY",
    "FEATURE_COUNT",
    "FEATURE_LABEL_COLUMN",
    "MIN_PACKETS_PER_WINDOW",
    "RAW_COLUMNS",
    "TIME_BIN_SECONDS",
    "WINDOW_SECONDS",
    "Application",
    "CaptureMetadata",
    "TrafficCategory",
    "VnatDatasetError",
    "VnatDatasetSummary",
    "VnatFilenameError",
    "VnatInspectionReport",
    "VpnStatus",
    "inspect_raw_dataframe",
    "inspect_vnat_file",
    "parse_capture_filename",
]
