"""Application bootstrap for the Parallax operator API."""

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Final

from fastapi import FastAPI

from parallax.features import (
    RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256,
    RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SHA256,
)
from parallax.modeling.runtime import load_prototype_runtime
from parallax.operator.api import create_operator_app
from parallax.operator.history import SqliteOperatorHistory
from parallax.operator.live_runtime import OperatorLiveRuntimeExecutor
from parallax.operator.service import OperatorModelIdentity, OperatorReplayService

ACCEPTED_PROTOTYPE_MODEL_SHA256: Final = (
    "1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7"
)
ACCEPTED_PROTOTYPE_CALIBRATION_SHA256: Final = (
    "af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d"
)

_DEFAULT_CAPTURE_ROOT = Path("data/raw/vnat/selected-pcaps")
_DEFAULT_MODEL_BUNDLE = Path("data/processed/vnat-release-1/prototype-model.json")
_DEFAULT_CALIBRATION_ARTIFACT = Path("data/processed/vnat-release-1/prototype-ood-calibration.json")


_DEFAULT_HISTORY_DATABASE = Path("data/operator/parallax-operator.sqlite3")


@dataclass(frozen=True, slots=True)
class OperatorApplicationConfig:
    """Filesystem configuration for the local operator application."""

    capture_root: Path
    model_bundle: Path
    calibration_artifact: Path
    history_database: Path

    @classmethod
    def from_environment(cls) -> "OperatorApplicationConfig":
        """Resolve configurable file locations without weakening artifact identity."""
        return cls(
            capture_root=Path(
                environ.get(
                    "PARALLAX_CAPTURE_ROOT",
                    str(_DEFAULT_CAPTURE_ROOT),
                )
            ),
            model_bundle=Path(
                environ.get(
                    "PARALLAX_MODEL_BUNDLE",
                    str(_DEFAULT_MODEL_BUNDLE),
                )
            ),
            calibration_artifact=Path(
                environ.get(
                    "PARALLAX_CALIBRATION_ARTIFACT",
                    str(_DEFAULT_CALIBRATION_ARTIFACT),
                )
            ),
            history_database=Path(
                environ.get(
                    "PARALLAX_HISTORY_DATABASE",
                    str(_DEFAULT_HISTORY_DATABASE),
                )
            ),
        )


def create_operator_application(
    config: OperatorApplicationConfig | None = None,
) -> FastAPI:
    """Load the accepted frozen runtime and create the operator API."""
    resolved = OperatorApplicationConfig.from_environment() if config is None else config

    runtime = load_prototype_runtime(
        resolved.model_bundle,
        resolved.calibration_artifact,
        expected_model_bundle_sha256=ACCEPTED_PROTOTYPE_MODEL_SHA256,
        expected_calibration_artifact_sha256=(ACCEPTED_PROTOTYPE_CALIBRATION_SHA256),
        expected_feature_artifact_sha256=(RELEASE_COMPATIBLE_FEATURE_ARTIFACT_SHA256),
        expected_split_manifest_sha256=(RELEASE_COMPATIBLE_CAPTURE_SPLIT_MANIFEST_SHA256),
    )

    service = OperatorReplayService(
        capture_root=resolved.capture_root,
        scorer=runtime,
        model_identity=OperatorModelIdentity.from_runtime(runtime),
        history=SqliteOperatorHistory(resolved.history_database),
        live_executor=OperatorLiveRuntimeExecutor(runtime),
    )

    return create_operator_app(service)
