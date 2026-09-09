"""Versioned metadata-only IPC protocol for the privileged Parallax sensor."""

import json
from dataclasses import dataclass
from math import isfinite
from typing import Final, cast

from parallax.data import PacketMetadata

SENSOR_IPC_SCHEMA_VERSION: Final = "parallax-sensor-ipc-1"
SENSOR_IPC_MAX_MESSAGE_BYTES: Final = 4_096


class SensorIpcProtocolError(ValueError):
    """Raised when a sensor IPC message violates the wire contract."""


@dataclass(frozen=True, slots=True)
class SensorStartRequest:
    """Request that the privileged sensor open one local interface."""

    interface: str


@dataclass(frozen=True, slots=True)
class SensorReadyMessage:
    """Acknowledgement that the requested capture source is open."""

    interface: str


@dataclass(frozen=True, slots=True)
class SensorPacketMessage:
    """One metadata-only packet emitted by the privileged sensor."""

    packet: PacketMetadata


@dataclass(frozen=True, slots=True)
class SensorErrorMessage:
    """Structured sensor failure sent before closing the IPC stream."""

    code: str
    message: str


SensorIpcMessage = (
    SensorStartRequest | SensorReadyMessage | SensorPacketMessage | SensorErrorMessage
)


def encode_sensor_ipc_message(
    message: SensorIpcMessage,
) -> bytes:
    """Encode exactly one bounded newline-delimited JSON message."""
    payload = _message_payload(message)

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SensorIpcProtocolError("sensor IPC message is not valid JSON data") from error

    framed = encoded + b"\n"

    if len(framed) > SENSOR_IPC_MAX_MESSAGE_BYTES:
        raise SensorIpcProtocolError("sensor IPC message exceeds the maximum size")

    return framed


def decode_sensor_ipc_message(
    framed: bytes,
) -> SensorIpcMessage:
    """Decode and strictly validate exactly one framed IPC message."""
    if not framed:
        raise SensorIpcProtocolError("sensor IPC message must not be empty")

    if len(framed) > SENSOR_IPC_MAX_MESSAGE_BYTES:
        raise SensorIpcProtocolError("sensor IPC message exceeds the maximum size")

    if not framed.endswith(b"\n"):
        raise SensorIpcProtocolError("sensor IPC message must end with a newline")

    if b"\n" in framed[:-1]:
        raise SensorIpcProtocolError("sensor IPC decoder accepts exactly one message")

    try:
        decoded = framed[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SensorIpcProtocolError("sensor IPC message must be UTF-8") from error

    try:
        raw: object = json.loads(decoded)
    except json.JSONDecodeError as error:
        raise SensorIpcProtocolError("sensor IPC message must contain valid JSON") from error

    payload = _require_object(
        raw,
        description="sensor IPC message",
    )

    schema_version = _require_string(
        payload,
        "schema_version",
    )

    if schema_version != SENSOR_IPC_SCHEMA_VERSION:
        raise SensorIpcProtocolError(f"unsupported sensor IPC schema version: {schema_version!r}")

    message_type = _require_string(
        payload,
        "type",
    )

    if message_type == "start":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "type",
                "interface",
            },
        )

        return SensorStartRequest(
            interface=_require_nonempty_string(
                payload,
                "interface",
            )
        )

    if message_type == "ready":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "type",
                "interface",
            },
        )

        return SensorReadyMessage(
            interface=_require_nonempty_string(
                payload,
                "interface",
            )
        )

    if message_type == "packet":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "type",
                "packet",
            },
        )

        packet_payload = _require_object(
            payload["packet"],
            description="sensor packet",
        )

        return SensorPacketMessage(packet=_decode_packet(packet_payload))

    if message_type == "error":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "type",
                "code",
                "message",
            },
        )

        return SensorErrorMessage(
            code=_require_nonempty_string(
                payload,
                "code",
            ),
            message=_require_nonempty_string(
                payload,
                "message",
            ),
        )

    raise SensorIpcProtocolError(f"unsupported sensor IPC message type: {message_type!r}")


def _message_payload(
    message: SensorIpcMessage,
) -> dict[str, object]:
    if isinstance(message, SensorStartRequest):
        return {
            "schema_version": SENSOR_IPC_SCHEMA_VERSION,
            "type": "start",
            "interface": _validated_text(
                message.interface,
                description="capture interface",
            ),
        }

    if isinstance(message, SensorReadyMessage):
        return {
            "schema_version": SENSOR_IPC_SCHEMA_VERSION,
            "type": "ready",
            "interface": _validated_text(
                message.interface,
                description="capture interface",
            ),
        }

    if isinstance(message, SensorPacketMessage):
        return {
            "schema_version": SENSOR_IPC_SCHEMA_VERSION,
            "type": "packet",
            "packet": _packet_payload(message.packet),
        }

    if isinstance(message, SensorErrorMessage):
        return {
            "schema_version": SENSOR_IPC_SCHEMA_VERSION,
            "type": "error",
            "code": _validated_text(
                message.code,
                description="sensor error code",
            ),
            "message": _validated_text(
                message.message,
                description="sensor error message",
            ),
        }

    raise SensorIpcProtocolError("unsupported sensor IPC message object")


def _packet_payload(
    packet: PacketMetadata,
) -> dict[str, object]:
    timestamp = float(packet.timestamp_seconds)

    if not isfinite(timestamp):
        raise SensorIpcProtocolError("packet timestamp must be finite")

    source_address = _validated_text(
        packet.source_address,
        description="packet source address",
    )
    destination_address = _validated_text(
        packet.destination_address,
        description="packet destination address",
    )

    _validate_integer_range(
        packet.source_port,
        description="packet source port",
        minimum=0,
        maximum=65_535,
    )
    _validate_integer_range(
        packet.destination_port,
        description="packet destination port",
        minimum=0,
        maximum=65_535,
    )
    _validate_integer_range(
        packet.protocol,
        description="packet protocol",
        minimum=0,
        maximum=255,
    )
    _validate_integer_range(
        packet.size,
        description="packet size",
        minimum=1,
        maximum=None,
    )

    return {
        "timestamp_seconds": timestamp,
        "source_address": source_address,
        "source_port": packet.source_port,
        "destination_address": destination_address,
        "destination_port": packet.destination_port,
        "protocol": packet.protocol,
        "size": packet.size,
    }


def _decode_packet(
    payload: dict[str, object],
) -> PacketMetadata:
    _require_exact_keys(
        payload,
        {
            "timestamp_seconds",
            "source_address",
            "source_port",
            "destination_address",
            "destination_port",
            "protocol",
            "size",
        },
    )

    timestamp = _require_number(
        payload,
        "timestamp_seconds",
    )

    if not isfinite(timestamp):
        raise SensorIpcProtocolError("packet timestamp must be finite")

    source_port = _require_integer(
        payload,
        "source_port",
    )
    destination_port = _require_integer(
        payload,
        "destination_port",
    )
    protocol = _require_integer(
        payload,
        "protocol",
    )
    size = _require_integer(
        payload,
        "size",
    )

    _validate_integer_range(
        source_port,
        description="packet source port",
        minimum=0,
        maximum=65_535,
    )
    _validate_integer_range(
        destination_port,
        description="packet destination port",
        minimum=0,
        maximum=65_535,
    )
    _validate_integer_range(
        protocol,
        description="packet protocol",
        minimum=0,
        maximum=255,
    )
    _validate_integer_range(
        size,
        description="packet size",
        minimum=1,
        maximum=None,
    )

    return PacketMetadata(
        timestamp_seconds=timestamp,
        source_address=_require_nonempty_string(
            payload,
            "source_address",
        ),
        source_port=source_port,
        destination_address=_require_nonempty_string(
            payload,
            "destination_address",
        ),
        destination_port=destination_port,
        protocol=protocol,
        size=size,
    )


def _require_object(
    value: object,
    *,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SensorIpcProtocolError(f"{description} must be a JSON object")

    # JSON object keys are strings by definition.
    return cast(dict[str, object], value)


def _require_exact_keys(
    payload: dict[str, object],
    expected: set[str],
) -> None:
    actual = set(payload)

    if actual != expected:
        raise SensorIpcProtocolError("sensor IPC message fields do not match the protocol")


def _require_string(
    payload: dict[str, object],
    key: str,
) -> str:
    value = payload.get(key)

    if not isinstance(value, str):
        raise SensorIpcProtocolError(f"sensor IPC field {key!r} must be a string")

    return value


def _require_nonempty_string(
    payload: dict[str, object],
    key: str,
) -> str:
    return _validated_text(
        _require_string(payload, key),
        description=f"sensor IPC field {key!r}",
    )


def _validated_text(
    value: str,
    *,
    description: str,
) -> str:
    if not value.strip():
        raise SensorIpcProtocolError(f"{description} must not be empty")

    return value


def _require_integer(
    payload: dict[str, object],
    key: str,
) -> int:
    value = payload.get(key)

    if type(value) is not int:
        raise SensorIpcProtocolError(f"sensor IPC field {key!r} must be an integer")

    return value


def _require_number(
    payload: dict[str, object],
    key: str,
) -> float:
    value = payload.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise SensorIpcProtocolError(f"sensor IPC field {key!r} must be numeric")

    return float(value)


def _validate_integer_range(
    value: int,
    *,
    description: str,
    minimum: int,
    maximum: int | None,
) -> None:
    if type(value) is not int:
        raise SensorIpcProtocolError(f"{description} must be an integer")

    if value < minimum:
        raise SensorIpcProtocolError(f"{description} is outside the allowed range")

    if maximum is not None and value > maximum:
        raise SensorIpcProtocolError(f"{description} is outside the allowed range")
