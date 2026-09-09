from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

import parallax.operator.application as application
from parallax.modeling.runtime import PrototypeRuntime
from parallax.operator import (
    ACCEPTED_PROTOTYPE_CALIBRATION_SHA256,
    ACCEPTED_PROTOTYPE_MODEL_SHA256,
    OperatorApplicationConfig,
    create_operator_application,
)

FEATURE_SHA256 = "611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16"
SPLIT_SHA256 = "a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f"


def _runtime() -> PrototypeRuntime:
    return cast(
        PrototypeRuntime,
        SimpleNamespace(
            bundle=SimpleNamespace(
                sha256=ACCEPTED_PROTOTYPE_MODEL_SHA256,
                feature_artifact_sha256=FEATURE_SHA256,
                split_manifest_sha256=SPLIT_SHA256,
            ),
            calibration=SimpleNamespace(
                sha256=ACCEPTED_PROTOTYPE_CALIBRATION_SHA256,
            ),
        ),
    )


def _install_runtime_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, Path, str, str, str, str]]:
    calls: list[tuple[Path, Path, str, str, str, str]] = []

    def fake_load_prototype_runtime(
        model_bundle: str | Path,
        calibration_artifact: str | Path,
        *,
        expected_model_bundle_sha256: str,
        expected_calibration_artifact_sha256: str,
        expected_feature_artifact_sha256: str,
        expected_split_manifest_sha256: str,
    ) -> PrototypeRuntime:
        calls.append(
            (
                Path(model_bundle),
                Path(calibration_artifact),
                expected_model_bundle_sha256,
                expected_calibration_artifact_sha256,
                expected_feature_artifact_sha256,
                expected_split_manifest_sha256,
            )
        )
        return _runtime()

    monkeypatch.setattr(
        application,
        "load_prototype_runtime",
        fake_load_prototype_runtime,
    )

    return calls


def test_environment_configuration_uses_repository_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARALLAX_CAPTURE_ROOT", raising=False)
    monkeypatch.delenv("PARALLAX_MODEL_BUNDLE", raising=False)
    monkeypatch.delenv("PARALLAX_CALIBRATION_ARTIFACT", raising=False)

    config = OperatorApplicationConfig.from_environment()

    assert config == OperatorApplicationConfig(
        capture_root=Path("data/raw/vnat/selected-pcaps"),
        model_bundle=Path("data/processed/vnat-release-1/prototype-model.json"),
        calibration_artifact=Path("data/processed/vnat-release-1/prototype-ood-calibration.json"),
    )


def test_environment_configuration_accepts_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARALLAX_CAPTURE_ROOT", "/captures")
    monkeypatch.setenv("PARALLAX_MODEL_BUNDLE", "/models/model.json")
    monkeypatch.setenv(
        "PARALLAX_CALIBRATION_ARTIFACT",
        "/models/calibration.json",
    )

    assert OperatorApplicationConfig.from_environment() == (
        OperatorApplicationConfig(
            capture_root=Path("/captures"),
            model_bundle=Path("/models/model.json"),
            calibration_artifact=Path("/models/calibration.json"),
        )
    )


def test_application_binds_explicit_paths_to_accepted_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_runtime_loader(monkeypatch)

    config = OperatorApplicationConfig(
        capture_root=tmp_path,
        model_bundle=tmp_path / "model.json",
        calibration_artifact=tmp_path / "calibration.json",
    )

    app = create_operator_application(config)

    assert app.title == "Parallax Operator API"
    assert calls == [
        (
            config.model_bundle,
            config.calibration_artifact,
            ACCEPTED_PROTOTYPE_MODEL_SHA256,
            ACCEPTED_PROTOTYPE_CALIBRATION_SHA256,
            FEATURE_SHA256,
            SPLIT_SHA256,
        )
    ]

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["active_model"] == {
        "model_bundle_sha256": ACCEPTED_PROTOTYPE_MODEL_SHA256,
        "calibration_artifact_sha256": (ACCEPTED_PROTOTYPE_CALIBRATION_SHA256),
        "feature_artifact_sha256": FEATURE_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
    }


def test_application_can_boot_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_runtime_loader(monkeypatch)

    monkeypatch.setenv("PARALLAX_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "PARALLAX_MODEL_BUNDLE",
        str(tmp_path / "model.json"),
    )
    monkeypatch.setenv(
        "PARALLAX_CALIBRATION_ARTIFACT",
        str(tmp_path / "calibration.json"),
    )

    app = create_operator_application()

    assert app.title == "Parallax Operator API"
    assert len(calls) == 1
    assert calls[0][0] == tmp_path / "model.json"
    assert calls[0][1] == tmp_path / "calibration.json"
