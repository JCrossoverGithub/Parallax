"""Lightweight bidirectional-flow assignment for long-running runtime use."""

from math import isfinite

from parallax.data.flows import (
    FlowConstructionError,
    FlowPacketAssignment,
)
from parallax.data.packets import PacketMetadata
from parallax.data.windowing import ConnectionKey


class RuntimeFlowTracker:
    """Assign runtime packets without retaining historical packet arrays."""

    __slots__ = (
        "_last_seen_by_connection",
        "_packet_number",
        "_previous_timestamp",
    )

    def __init__(self) -> None:
        self._last_seen_by_connection: dict[ConnectionKey, float] = {}
        self._previous_timestamp: float | None = None
        self._packet_number = 0

    @property
    def packet_count(self) -> int:
        """Return the number of packets assigned by this tracker."""
        return self._packet_number

    @property
    def tracked_flow_count(self) -> int:
        """Return the number of bidirectional flow identities retained."""
        return len(self._last_seen_by_connection)

    def push(self, packet: PacketMetadata) -> FlowPacketAssignment:
        """Validate and assign one packet to a canonical runtime flow."""
        packet_number = self._packet_number + 1
        timestamp = packet.timestamp_seconds

        if not isfinite(timestamp):
            raise FlowConstructionError(f"packet {packet_number}: timestamp must be finite")

        if self._previous_timestamp is not None and timestamp < self._previous_timestamp:
            raise FlowConstructionError(
                f"packet {packet_number}: timestamp precedes the previous packet"
            )

        if packet.size <= 0:
            raise FlowConstructionError(f"packet {packet_number}: size must be greater than zero")

        forward = packet.connection
        reverse = _reverse_connection(forward)

        if forward in self._last_seen_by_connection:
            connection = forward
            direction = 1
        elif reverse in self._last_seen_by_connection:
            connection = reverse
            direction = 0
        else:
            connection = forward
            direction = 1

        self._last_seen_by_connection[connection] = timestamp
        self._previous_timestamp = timestamp
        self._packet_number = packet_number

        return FlowPacketAssignment(
            packet_number=packet_number,
            connection=connection,
            direction=direction,
            packet=packet,
        )


def _reverse_connection(
    connection: ConnectionKey,
) -> ConnectionKey:
    source, source_port, destination, destination_port, protocol = connection

    return (
        destination,
        destination_port,
        source,
        source_port,
        protocol,
    )
