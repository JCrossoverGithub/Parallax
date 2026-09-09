from collections.abc import Sequence
from pathlib import Path
from socket import inet_aton
from time import monotonic, sleep
from types import SimpleNamespace
from typing import cast

import dpkt  # type: ignore[import-untyped]
import pytest

from parallax.data import FEATURE_COUNT, IP_PROTOCOL_TCP, PCAP_LINKTYPE_RAW_IP
from parallax.features import VnatWindowFeature
from parallax.modeling.baselines import CATEGORY_LABELS
from parallax.modeling.runtime import PrototypeRuntime, PrototypeRuntimePrediction
from parallax.operator import (
    OperatorModelIdentity,
    OperatorReplayNotFoundError,
    OperatorReplayService,
    OperatorServiceError,
)
from parallax.replay import ReplayState

SOURCE = "10.101.1.100"
DESTINATION = "10.103.1.100"


class FakeScorer:
    def score_feature(
        self,
        feature: VnatWindowFeature,
    ) -> PrototypeRuntimePrediction:
        assert feature.values.shape == (FEATURE_COUNT,)
        probabilities = (0.7, 0.1, 0.1, 0.05, 0.05)

        return PrototypeRuntimePrediction(
            window_id=feature.window_id,
            capture_id=feature.capture.capture_id,
            flow_id=feature.flow_id,
            window_index=feature.window_index,
            start_offset_seconds=feature.start_offset_seconds,
            end_offset_seconds=feature.end_offset_seconds,
            packet_count=feature.packet_count,
            category_order=CATEGORY_LABELS,
            class_probabilities=probabilities,
            predicted_class_index=0,
            predicted_category=CATEGORY_LABELS[0],
            raw_confidence=probabilities[0],
            relative_mahalanobis_distance=1.0,
            ood_score=0.1,
            model_bundle_sha256="a" * 64,
            calibration_artifact_sha256="b" * 64,
            feature_artifact_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
        )


def _identity() -> OperatorModelIdentity:
    return OperatorModelIdentity(
        model_bundle_sha256="a" * 64,
        calibration_artifact_sha256="b" * 64,
        feature_artifact_sha256="c" * 64,
        split_manifest_sha256="d" * 64,
    )


def _ipv4_packet() -> bytes:
    transport = dpkt.tcp.TCP(
        sport=41_898,
        dport=22,
        flags=dpkt.tcp.TH_ACK,
        data=b"encrypted",
    )
    packet = dpkt.ip.IP(
        src=inet_aton(SOURCE),
        dst=inet_aton(DESTINATION),
        p=IP_PROTOCOL_TCP,
        ttl=64,
        data=transport,
    )
    packet.len = len(packet)
    return bytes(packet)


def _write_pcap(
    path: Path,
    records: Sequence[tuple[float, bytes]],
) -> None:
    with path.open("wb") as stream:
        writer = dpkt.pcap.Writer(
            stream,
            linktype=PCAP_LINKTYPE_RAW_IP,
        )

        for timestamp, packet in records:
            writer.writepkt(packet, ts=timestamp)

        writer.close()


def _service(
    capture_root: Path,
    *,
    event_history_limit: int = 10_000,
) -> OperatorReplayService:
    return OperatorReplayService(
        capture_root=capture_root,
        scorer=FakeScorer(),
        model_identity=_identity(),
        event_history_limit=event_history_limit,
    )


def _wait_for_terminal(
    service: OperatorReplayService,
    run_id: str,
) -> None:
    deadline = monotonic() + 2.0

    while monotonic() < deadline:
        if service.get_replay(run_id).state in {
            ReplayState.COMPLETED,
            ReplayState.FAILED,
            ReplayState.CANCELLED,
        }:
            return

        sleep(0.01)

    raise AssertionError("replay did not reach a terminal state")


def test_runs_real_packet_pipeline_and_retains_prediction_events(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture91.pcap"
    _write_pcap(
        source,
        [(100.0 + index * 0.1, _ipv4_packet()) for index in range(21)],
    )
    service = _service(tmp_path)

    created = service.start_replay(
        source.name,
        time_scale=None,
    )

    assert created.state is ReplayState.CREATED
    assert created.source_id == source.name
    assert created.event_count == 0

    _wait_for_terminal(service, created.run_id)

    completed = service.get_replay(created.run_id)
    events = service.get_events(created.run_id)

    assert completed.state is ReplayState.COMPLETED
    assert completed.event_count == 1
    assert completed.retained_event_count == 1
    assert completed.as_dict()["failure"] is None
    assert len(events) == 1
    assert events[0]["run_id"] == created.run_id
    window = events[0]["window"]
    assert isinstance(window, dict)
    assert window["capture_id"] == source.name


def test_bounds_retained_event_history(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture92.pcap"
    records: list[tuple[float, bytes]] = []

    for window_index in range(3):
        start = 100.0 + (window_index * 41.0)

        records.extend((start + index * 0.1, _ipv4_packet()) for index in range(21))

    _write_pcap(source, records)
    service = _service(tmp_path, event_history_limit=2)

    created = service.start_replay(source.name, time_scale=None)
    _wait_for_terminal(service, created.run_id)

    snapshot = service.get_replay(created.run_id)
    events = service.get_events(created.run_id)

    assert snapshot.event_count == 3
    assert snapshot.retained_event_count == 2
    assert len(events) == 2


def test_health_reports_active_model_and_session_counts(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.health() == {
        "status": "ok",
        "active_model": _identity().as_dict(),
        "sessions": {
            "total": 0,
            "active": 0,
        },
    }


@pytest.mark.parametrize(
    "capture_name",
    [
        "",
        "../nonvpn_ssh_capture1.pcap",
    ],
)
def test_rejects_non_filename_capture_identifiers(
    tmp_path: Path,
    capture_name: str,
) -> None:
    with pytest.raises(OperatorServiceError, match="filename only"):
        _service(tmp_path).start_replay(capture_name)


def test_rejects_missing_capture(tmp_path: Path) -> None:
    with pytest.raises(OperatorServiceError, match="does not exist"):
        _service(tmp_path).start_replay("nonvpn_ssh_capture404.pcap")


def test_rejects_invalid_event_history_limit(tmp_path: Path) -> None:
    with pytest.raises(OperatorServiceError, match="history limit must be positive"):
        _service(tmp_path, event_history_limit=0)


def test_rejects_unknown_replay_id(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(OperatorReplayNotFoundError, match="does not exist"):
        service.get_replay("missing")

    with pytest.raises(OperatorReplayNotFoundError, match="does not exist"):
        service.get_events("missing")


def test_builds_operator_identity_from_verified_runtime() -> None:
    runtime = cast(
        PrototypeRuntime,
        SimpleNamespace(
            bundle=SimpleNamespace(
                sha256="a" * 64,
                feature_artifact_sha256="c" * 64,
                split_manifest_sha256="d" * 64,
            ),
            calibration=SimpleNamespace(
                sha256="b" * 64,
            ),
        ),
    )

    assert OperatorModelIdentity.from_runtime(runtime) == _identity()


def test_rejects_invalid_replay_configuration(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture93.pcap"
    _write_pcap(source, [])

    with pytest.raises(OperatorServiceError, match="greater than zero"):
        _service(tmp_path).start_replay(
            source.name,
            time_scale=0.0,
        )


def test_failed_replay_exposes_structured_failure(tmp_path: Path) -> None:
    source = tmp_path / "nonvpn_ssh_capture94.pcap"

    # A classic Raw-IP PCAP containing an IPv6 record is deliberately
    # unsupported by the VNAT metadata parser.
    _write_pcap(
        source,
        [
            (
                100.0,
                b"\x60" + b"\x00" * 39,
            )
        ],
    )

    service = _service(tmp_path)
    created = service.start_replay(
        source.name,
        time_scale=None,
    )

    _wait_for_terminal(service, created.run_id)

    failed = service.get_replay(created.run_id)
    payload = failed.as_dict()

    assert failed.state is ReplayState.FAILED

    failure = payload["failure"]
    assert isinstance(failure, dict)
    assert failure["code"] == "replay_execution_error"
    assert "unsupported IP version" in failure["message"]


def _wait_for_state(
    service: OperatorReplayService,
    run_id: str,
    state: ReplayState,
) -> None:
    deadline = monotonic() + 2.0

    while monotonic() < deadline:
        if service.get_replay(run_id).state is state:
            return

        sleep(0.01)

    raise AssertionError(f"replay did not reach {state.value!r}")


def test_pause_resume_and_cancel_control_real_replay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture95.pcap"
    _write_pcap(
        source,
        [
            (100.0, _ipv4_packet()),
            (110.0, _ipv4_packet()),
        ],
    )

    service = _service(tmp_path)
    created = service.start_replay(
        source.name,
        time_scale=1.0,
    )

    _wait_for_state(
        service,
        created.run_id,
        ReplayState.RUNNING,
    )

    service.pause_replay(created.run_id)

    _wait_for_state(
        service,
        created.run_id,
        ReplayState.PAUSED,
    )

    service.resume_replay(created.run_id)

    _wait_for_state(
        service,
        created.run_id,
        ReplayState.RUNNING,
    )

    service.cancel_replay(created.run_id)
    _wait_for_terminal(service, created.run_id)

    assert service.get_replay(created.run_id).state is ReplayState.CANCELLED


def test_terminal_replay_rejects_control_request(
    tmp_path: Path,
) -> None:
    from parallax.operator import OperatorReplayConflictError

    source = tmp_path / "nonvpn_ssh_capture96.pcap"
    _write_pcap(source, [])

    service = _service(tmp_path)
    created = service.start_replay(
        source.name,
        time_scale=None,
    )
    _wait_for_terminal(service, created.run_id)

    with pytest.raises(
        OperatorReplayConflictError,
        match="already terminal",
    ):
        service.pause_replay(created.run_id)


def test_event_batch_sequences_retained_history(
    tmp_path: Path,
) -> None:
    from parallax.operator import OperatorEventCursorError

    source = tmp_path / "nonvpn_ssh_capture97.pcap"
    records: list[tuple[float, bytes]] = []

    for window_index in range(3):
        start = 100.0 + (window_index * 41.0)

        records.extend((start + index * 0.1, _ipv4_packet()) for index in range(21))

    _write_pcap(source, records)

    service = _service(
        tmp_path,
        event_history_limit=2,
    )

    created = service.start_replay(
        source.name,
        time_scale=None,
    )
    _wait_for_terminal(service, created.run_id)

    batch = service.get_event_batch(
        created.run_id,
        after_sequence=1,
    )

    assert [event.sequence for event in batch.events] == [2, 3]
    assert batch.last_sequence == 3
    assert batch.state is ReplayState.COMPLETED

    with pytest.raises(
        OperatorEventCursorError,
        match="no longer retained",
    ):
        service.get_event_batch(
            created.run_id,
            after_sequence=0,
        )

    with pytest.raises(
        OperatorEventCursorError,
        match="ahead of the replay",
    ):
        service.get_event_batch(
            created.run_id,
            after_sequence=4,
        )

    with pytest.raises(
        OperatorEventCursorError,
        match="must not be negative",
    ):
        service.get_event_batch(
            created.run_id,
            after_sequence=-1,
        )


def test_empty_replay_has_empty_event_batch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonvpn_ssh_capture98.pcap"
    _write_pcap(source, [])

    service = _service(tmp_path)
    created = service.start_replay(
        source.name,
        time_scale=None,
    )
    _wait_for_terminal(service, created.run_id)

    batch = service.get_event_batch(created.run_id)

    assert batch.events == ()
    assert batch.last_sequence == 0
    assert batch.state is ReplayState.COMPLETED


def test_control_error_becomes_operator_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from parallax.operator import OperatorReplayConflictError
    from parallax.replay import ReplayControl, ReplayControlError

    source = tmp_path / "nonvpn_ssh_capture99.pcap"
    _write_pcap(
        source,
        [
            (100.0, _ipv4_packet()),
            (110.0, _ipv4_packet()),
        ],
    )

    service = _service(tmp_path)
    created = service.start_replay(
        source.name,
        time_scale=1.0,
    )

    _wait_for_state(
        service,
        created.run_id,
        ReplayState.RUNNING,
    )

    def fail_pause(_: ReplayControl) -> None:
        raise ReplayControlError("synthetic control conflict")

    monkeypatch.setattr(
        ReplayControl,
        "pause",
        fail_pause,
    )

    with pytest.raises(
        OperatorReplayConflictError,
        match="synthetic control conflict",
    ):
        service.pause_replay(created.run_id)

    service.cancel_replay(created.run_id)
    _wait_for_terminal(service, created.run_id)

    assert service.get_replay(created.run_id).state is ReplayState.CANCELLED


def test_service_persists_completed_replay_and_events(
    tmp_path: Path,
) -> None:
    from parallax.operator import SqliteOperatorHistory

    source = tmp_path / "nonvpn_ssh_capture100.pcap"
    _write_pcap(
        source,
        [(100.0 + index * 0.1, _ipv4_packet()) for index in range(21)],
    )

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=FakeScorer(),
        model_identity=_identity(),
        history=history,
    )

    created = service.start_replay(
        source.name,
        time_scale=None,
    )
    _wait_for_terminal(service, created.run_id)

    persisted = service.get_history_replay(created.run_id)
    events = service.get_history_events(created.run_id)

    assert persisted.state is ReplayState.COMPLETED
    assert persisted.event_count == 1
    assert len(events) == 1
    assert events[0]["run_id"] == created.run_id


def test_service_lists_persisted_history(
    tmp_path: Path,
) -> None:
    from parallax.operator import SqliteOperatorHistory

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=FakeScorer(),
        model_identity=_identity(),
        history=history,
    )

    first = tmp_path / "nonvpn_ssh_capture101.pcap"
    second = tmp_path / "nonvpn_ssh_capture102.pcap"

    _write_pcap(first, [])
    _write_pcap(second, [])

    run_one = service.start_replay(
        first.name,
        time_scale=None,
    )
    _wait_for_terminal(service, run_one.run_id)

    run_two = service.start_replay(
        second.name,
        time_scale=None,
    )
    _wait_for_terminal(service, run_two.run_id)

    assert [record.run_id for record in service.list_history()][:2] == [
        run_two.run_id,
        run_one.run_id,
    ]


def test_service_without_history_has_empty_history(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    assert service.list_history() == ()

    with pytest.raises(
        OperatorReplayNotFoundError,
        match="persisted replay",
    ):
        service.get_history_replay("missing")


def test_history_validation_errors_become_service_errors(
    tmp_path: Path,
) -> None:
    from parallax.operator.history import SqliteOperatorHistory
    from parallax.operator.service import OperatorServiceError

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=FakeScorer(),
        model_identity=_identity(),
        history=history,
    )

    with pytest.raises(
        OperatorServiceError,
        match="history list limit must be positive",
    ):
        service.list_history(limit=0)


def test_configured_history_rejects_unknown_replay(
    tmp_path: Path,
) -> None:
    from parallax.operator.history import SqliteOperatorHistory
    from parallax.operator.service import (
        OperatorReplayNotFoundError,
    )

    history = SqliteOperatorHistory(tmp_path / "operator.sqlite3")

    service = OperatorReplayService(
        capture_root=tmp_path,
        scorer=FakeScorer(),
        model_identity=_identity(),
        history=history,
    )

    with pytest.raises(
        OperatorReplayNotFoundError,
        match="persisted replay",
    ):
        service.get_history_replay("missing")

    with pytest.raises(
        OperatorReplayNotFoundError,
        match="persisted replay",
    ):
        service.get_history_events("missing")
