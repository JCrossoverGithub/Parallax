"""Lightweight bounded bidirectional-flow assignment for runtime use."""

from dataclasses import dataclass
from math import isfinite

from parallax.data.flows import (
    FlowConstructionError,
    FlowPacketAssignment,
)
from parallax.data.packets import PacketMetadata
from parallax.data.windowing import ConnectionKey


class RuntimeFlowCapacityError(RuntimeError):
    """Raised when a new runtime flow would exceed configured capacity."""


@dataclass(frozen=True, slots=True)
class RuntimeFlowTrackerConfig:
    """Resource bounds for long-running runtime flow identity tracking."""

    stale_after_seconds: float | None = None
    max_tracked_flows: int | None = None

    def __post_init__(self) -> None:
        if self.stale_after_seconds is not None and (
            not isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0.0
        ):
            raise ValueError("stale flow timeout must be finite and positive")

        if self.max_tracked_flows is not None and self.max_tracked_flows < 1:
            raise ValueError("maximum tracked flows must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeFlowTrackerStats:
    """Immutable operational counters for one runtime flow tracker."""

    packet_count: int
    tracked_flow_count: int
    flows_created: int
    flows_evicted_stale: int
    capacity_rejections: int
    peak_tracked_flow_count: int


class RuntimeFlowTracker:
    """Assign runtime packets while retaining bounded flow identity state."""

    __slots__ = (
        "_capacity_rejections",
        "_config",
        "_flows_created",
        "_flows_evicted_stale",
        "_last_seen_by_connection",
        "_packet_number",
        "_peak_tracked_flow_count",
        "_previous_timestamp",
    )

    def __init__(
        self,
        *,
        config: RuntimeFlowTrackerConfig | None = None,
    ) -> None:
        self._config = config if config is not None else RuntimeFlowTrackerConfig()
        self._last_seen_by_connection: dict[ConnectionKey, float] = {}
        self._previous_timestamp: float | None = None
        self._packet_number = 0
        self._flows_created = 0
        self._flows_evicted_stale = 0
        self._capacity_rejections = 0
        self._peak_tracked_flow_count = 0

    @property
    def packet_count(self) -> int:
        """Return the number of successfully assigned packets."""
        return self._packet_number

    @property
    def tracked_flow_count(self) -> int:
        """Return the number of bidirectional flow identities retained."""
        return len(self._last_seen_by_connection)

    @property
    def stats(self) -> RuntimeFlowTrackerStats:
        """Return immutable resource and lifecycle counters."""
        return RuntimeFlowTrackerStats(
            packet_count=self._packet_number,
            tracked_flow_count=len(self._last_seen_by_connection),
            flows_created=self._flows_created,
            flows_evicted_stale=self._flows_evicted_stale,
            capacity_rejections=self._capacity_rejections,
            peak_tracked_flow_count=self._peak_tracked_flow_count,
        )

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

        self._evict_stale(timestamp)

        forward = packet.connection
        reverse = _reverse_connection(forward)

        if forward in self._last_seen_by_connection:
            connection = forward
            direction = 1
        elif reverse in self._last_seen_by_connection:
            connection = reverse
            direction = 0
        else:
            self._ensure_capacity()
            connection = forward
            direction = 1
            self._flows_created += 1

        self._last_seen_by_connection[connection] = timestamp
        self._peak_tracked_flow_count = max(
            self._peak_tracked_flow_count,
            len(self._last_seen_by_connection),
        )
        self._previous_timestamp = timestamp
        self._packet_number = packet_number

        return FlowPacketAssignment(
            packet_number=packet_number,
            connection=connection,
            direction=direction,
            packet=packet,
        )

    def _evict_stale(self, timestamp: float) -> None:
        stale_after_seconds = self._config.stale_after_seconds

        if stale_after_seconds is None:
            return

        stale_connections = tuple(
            connection
            for connection, last_seen in self._last_seen_by_connection.items()
            if timestamp - last_seen >= stale_after_seconds
        )

        for connection in stale_connections:
            del self._last_seen_by_connection[connection]

        self._flows_evicted_stale += len(stale_connections)

    def _ensure_capacity(self) -> None:
        maximum = self._config.max_tracked_flows

        if maximum is None or len(self._last_seen_by_connection) < maximum:
            return

        self._capacity_rejections += 1
        raise RuntimeFlowCapacityError(f"runtime flow capacity reached: {maximum}")


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
