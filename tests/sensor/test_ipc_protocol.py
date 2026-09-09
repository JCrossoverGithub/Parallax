import json
from dataclasses import replace
from math import nan
from typing import Any, cast

import pytest

from parallax.data import PacketMetadata
from parallax.sensor.ipc_protocol import (
    SENSOR_IPC_MAX_MESSAGE_BYTES,
    SENSOR_IPC_SCHEMA_VERSION,
    SensorErrorMessage,
    SensorIpcProtocolError,
    SensorPacketMessage,
    SensorReadyMessage,
    SensorStartRequest,
    decode_sensor_ipc_message,
    encode_sensor_ipc_message,
)


def _packet() -> PacketMetadata:
    return PacketMetadata(
        timestamp_seconds=123.5,
        source_address="10.0.0.10",
        source_port=50_000,
        destination_address="10.0.0.20",
        destination_port=443,
        protocol=6,
        size=1200,
    )


@pytest.mark.parametrize(
    "message",
    [
        SensorStartRequest(interface="eth0"),
        SensorReadyMessage(interface="eth0"),
        SensorPacketMessage(packet=_packet()),
        SensorErrorMessage(
            code="capture_failed",
            message="capture could not be opened",
        ),
    ],
)
def test_round_trips_supported_messages(
    message: object,
) -> None:
    encoded = encode_sensor_ipc_message(cast(Any, message))

    assert encoded.endswith(b"\n")
    assert decode_sensor_ipc_message(encoded) == message


def test_packet_wire_message_contains_metadata_only() -> None:
    encoded = encode_sensor_ipc_message(SensorPacketMessage(packet=_packet()))

    payload = json.loads(encoded)

    assert payload == {
        "schema_version": SENSOR_IPC_SCHEMA_VERSION,
        "type": "packet",
        "packet": {
            "timestamp_seconds": 123.5,
            "source_address": "10.0.0.10",
            "source_port": 50_000,
            "destination_address": "10.0.0.20",
            "destination_port": 443,
            "protocol": 6,
            "size": 1200,
        },
    }

    assert "payload" not in payload["packet"]
    assert "data" not in payload["packet"]


@pytest.mark.parametrize(
    "message",
    [
        SensorStartRequest(interface=" "),
        SensorReadyMessage(interface=""),
        SensorErrorMessage(
            code="",
            message="failure",
        ),
        SensorErrorMessage(
            code="capture_failed",
            message="\t",
        ),
    ],
)
def test_rejects_blank_encoded_text_fields(
    message: object,
) -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="must not be empty",
    ):
        encode_sensor_ipc_message(cast(Any, message))


@pytest.mark.parametrize(
    ("packet", "match"),
    [
        (
            replace(
                _packet(),
                timestamp_seconds=nan,
            ),
            "timestamp must be finite",
        ),
        (
            replace(
                _packet(),
                source_address="",
            ),
            "source address must not be empty",
        ),
        (
            replace(
                _packet(),
                destination_address=" ",
            ),
            "destination address must not be empty",
        ),
        (
            replace(
                _packet(),
                source_port=-1,
            ),
            "source port is outside",
        ),
        (
            replace(
                _packet(),
                destination_port=65_536,
            ),
            "destination port is outside",
        ),
        (
            replace(
                _packet(),
                protocol=256,
            ),
            "protocol is outside",
        ),
        (
            replace(
                _packet(),
                size=0,
            ),
            "size is outside",
        ),
    ],
)
def test_rejects_invalid_encoded_packet_metadata(
    packet: PacketMetadata,
    match: str,
) -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match=match,
    ):
        encode_sensor_ipc_message(SensorPacketMessage(packet=packet))


def test_rejects_runtime_invalid_integer_packet_value() -> None:
    packet = replace(
        _packet(),
        source_port=cast(Any, True),
    )

    with pytest.raises(
        SensorIpcProtocolError,
        match="source port must be an integer",
    ):
        encode_sensor_ipc_message(SensorPacketMessage(packet=packet))


def test_rejects_unsupported_message_object() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="unsupported sensor IPC message object",
    ):
        encode_sensor_ipc_message(cast(Any, object()))


@pytest.mark.parametrize(
    ("framed", "match"),
    [
        (
            b"",
            "must not be empty",
        ),
        (
            b'{"type":"start"}',
            "must end with a newline",
        ),
        (
            b"{}\n{}\n",
            "exactly one message",
        ),
        (
            b"\xff\n",
            "must be UTF-8",
        ),
        (
            b"{not-json}\n",
            "must contain valid JSON",
        ),
        (
            b"[]\n",
            "must be a JSON object",
        ),
    ],
)
def test_rejects_invalid_message_framing(
    framed: bytes,
    match: str,
) -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match=match,
    ):
        decode_sensor_ipc_message(framed)


def test_rejects_oversized_message() -> None:
    framed = b"x" * SENSOR_IPC_MAX_MESSAGE_BYTES + b"\n"

    with pytest.raises(
        SensorIpcProtocolError,
        match="exceeds the maximum size",
    ):
        decode_sensor_ipc_message(framed)


def test_wraps_json_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_dumps(
        _value: object,
        *,
        allow_nan: bool,
        separators: tuple[str, str],
        sort_keys: bool,
    ) -> str:
        assert not allow_nan
        assert separators == (",", ":")
        assert sort_keys
        raise TypeError("serialization failed")

    monkeypatch.setattr(
        json,
        "dumps",
        failing_dumps,
    )

    with pytest.raises(
        SensorIpcProtocolError,
        match="sensor IPC message is not valid JSON data",
    ):
        encode_sensor_ipc_message(SensorStartRequest(interface="eth0"))


def test_rejects_oversized_encoded_message() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="sensor IPC message exceeds the maximum size",
    ):
        encode_sensor_ipc_message(
            SensorErrorMessage(
                code="capture_failed",
                message=("x" * SENSOR_IPC_MAX_MESSAGE_BYTES),
            )
        )


def _frame(
    payload: object,
) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def test_rejects_missing_schema_version() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match=r"schema_version.*must be a string",
    ):
        decode_sensor_ipc_message(
            _frame(
                {
                    "type": "start",
                    "interface": "eth0",
                }
            )
        )


def test_rejects_unknown_schema_version() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="unsupported sensor IPC schema version",
    ):
        decode_sensor_ipc_message(
            _frame(
                {
                    "schema_version": "future-version",
                    "type": "start",
                    "interface": "eth0",
                }
            )
        )


def test_rejects_unknown_message_type() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="unsupported sensor IPC message type",
    ):
        decode_sensor_ipc_message(
            _frame(
                {
                    "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
                    "type": "unknown",
                }
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
            "type": "start",
        },
        {
            "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
            "type": "ready",
            "interface": "eth0",
            "unexpected": True,
        },
        {
            "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
            "type": "error",
            "code": "failed",
            "message": "failure",
            "extra": "field",
        },
    ],
)
def test_rejects_message_field_drift(
    payload: dict[str, object],
) -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="fields do not match",
    ):
        decode_sensor_ipc_message(_frame(payload))


def test_rejects_non_object_packet_message() -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match="sensor packet must be a JSON object",
    ):
        decode_sensor_ipc_message(
            _frame(
                {
                    "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
                    "type": "packet",
                    "packet": [],
                }
            )
        )


def _packet_frame(
    **overrides: object,
) -> bytes:
    packet: dict[str, object] = {
        "timestamp_seconds": 123.5,
        "source_address": "10.0.0.10",
        "source_port": 50_000,
        "destination_address": "10.0.0.20",
        "destination_port": 443,
        "protocol": 6,
        "size": 1200,
    }

    packet.update(overrides)

    return _frame(
        {
            "schema_version": (SENSOR_IPC_SCHEMA_VERSION),
            "type": "packet",
            "packet": packet,
        }
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"timestamp_seconds": "bad"},
            "timestamp_seconds.*must be numeric",
        ),
        (
            {"timestamp_seconds": nan},
            "timestamp must be finite",
        ),
        (
            {"source_address": ""},
            "source_address.*must not be empty",
        ),
        (
            {"source_port": True},
            "source_port.*must be an integer",
        ),
        (
            {"source_port": -1},
            "source port is outside",
        ),
        (
            {"destination_port": 70_000},
            "destination port is outside",
        ),
        (
            {"protocol": 300},
            "protocol is outside",
        ),
        (
            {"size": 0},
            "size is outside",
        ),
    ],
)
def test_rejects_invalid_decoded_packet_metadata(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(
        SensorIpcProtocolError,
        match=match,
    ):
        decode_sensor_ipc_message(_packet_frame(**overrides))
