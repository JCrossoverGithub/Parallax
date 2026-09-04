"""Deterministic bidirectional-flow construction from packet metadata."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

from parallax.data.pcap import PacketMetadata
from parallax.data.windowing import ConnectionKey


class FlowConstructionError(ValueError):
    """Raised when packet metadata cannot form deterministic flows."""


@dataclass(frozen=True, slots=True)
class BidirectionalFlow:
    """Immutable packet metadata for one first-observed bidirectional flow."""

    connection: ConnectionKey
    timestamps: tuple[float, ...]
    sizes: tuple[int, ...]
    directions: tuple[int, ...]

    @property
    def packet_count(self) -> int:
        """Return the number of packets retained by the flow."""
        return len(self.timestamps)


@dataclass(slots=True)
class _FlowBuilder:
    connection: ConnectionKey
    timestamps: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)

    def append(self, packet: PacketMetadata, direction: int) -> None:
        self.timestamps.append(packet.timestamp_seconds)
        self.sizes.append(packet.size)
        self.directions.append(direction)

    def freeze(self) -> BidirectionalFlow:
        return BidirectionalFlow(
            connection=self.connection,
            timestamps=tuple(self.timestamps),
            sizes=tuple(self.sizes),
            directions=tuple(self.directions),
        )


def group_bidirectional_flows(
    packets: Iterable[PacketMetadata],
) -> tuple[BidirectionalFlow, ...]:
    """Group ordered packets using first observation as forward direction one."""
    builders: dict[ConnectionKey, _FlowBuilder] = {}
    previous_timestamp: float | None = None

    for packet_number, packet in enumerate(packets, start=1):
        timestamp = packet.timestamp_seconds
        if not isfinite(timestamp):
            raise FlowConstructionError(f"packet {packet_number}: timestamp must be finite")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise FlowConstructionError(
                f"packet {packet_number}: timestamp precedes the previous packet"
            )
        if packet.size <= 0:
            raise FlowConstructionError(f"packet {packet_number}: size must be greater than zero")
        previous_timestamp = timestamp

        forward = packet.connection
        reverse = _reverse_connection(forward)

        if forward in builders:
            key = forward
            direction = 1
        elif reverse in builders:
            key = reverse
            direction = 0
        else:
            key = forward
            direction = 1
            builders[key] = _FlowBuilder(connection=key)

        builders[key].append(packet, direction)

    return tuple(builder.freeze() for builder in builders.values())


def _reverse_connection(connection: ConnectionKey) -> ConnectionKey:
    source, source_port, destination, destination_port, protocol = connection
    return destination, destination_port, source, source_port, protocol
