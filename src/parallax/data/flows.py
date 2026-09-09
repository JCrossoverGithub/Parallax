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


@dataclass(frozen=True, slots=True)
class FlowPacketAssignment:
    """Canonical flow assignment produced for one incrementally observed packet."""

    packet_number: int
    connection: ConnectionKey
    direction: int
    packet: PacketMetadata


class BidirectionalFlowTracker:
    """Incrementally assign ordered packets to first-observed bidirectional flows."""

    __slots__ = ("_builders", "_packet_number", "_previous_timestamp")

    def __init__(self) -> None:
        self._builders: dict[ConnectionKey, _FlowBuilder] = {}
        self._previous_timestamp: float | None = None
        self._packet_number = 0

    def push(self, packet: PacketMetadata) -> FlowPacketAssignment:
        """Validate and append one packet, returning its canonical flow assignment."""
        packet_number = self._packet_number + 1
        timestamp = packet.timestamp_seconds

        if not isfinite(timestamp):
            raise FlowConstructionError(
                f"packet {packet_number}: timestamp must be finite"
            )
        if (
            self._previous_timestamp is not None
            and timestamp < self._previous_timestamp
        ):
            raise FlowConstructionError(
                f"packet {packet_number}: timestamp precedes the previous packet"
            )
        if packet.size <= 0:
            raise FlowConstructionError(
                f"packet {packet_number}: size must be greater than zero"
            )

        forward = packet.connection
        reverse = _reverse_connection(forward)

        if forward in self._builders:
            key = forward
            direction = 1
        elif reverse in self._builders:
            key = reverse
            direction = 0
        else:
            key = forward
            direction = 1
            self._builders[key] = _FlowBuilder(connection=key)

        self._builders[key].append(packet, direction)
        self._previous_timestamp = timestamp
        self._packet_number = packet_number

        return FlowPacketAssignment(
            packet_number=packet_number,
            connection=key,
            direction=direction,
            packet=packet,
        )

    def freeze(self) -> tuple[BidirectionalFlow, ...]:
        """Return immutable snapshots in first-observation flow order."""
        return tuple(builder.freeze() for builder in self._builders.values())


def group_bidirectional_flows(
    packets: Iterable[PacketMetadata],
) -> tuple[BidirectionalFlow, ...]:
    """Group ordered packets using first observation as forward direction one."""
    tracker = BidirectionalFlowTracker()

    for packet in packets:
        tracker.push(packet)

    return tracker.freeze()


def _reverse_connection(connection: ConnectionKey) -> ConnectionKey:
    source, source_port, destination, destination_port, protocol = connection
    return destination, destination_port, source, source_port, protocol
