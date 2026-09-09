"""Packet-aware replay entries produced from metadata-only PCAP parsing."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from parallax.data.pcap import (
    PacketMetadata,
    PacketSizePolicy,
    iter_pcap_packet_metadata,
)
from parallax.replay.domain import ReplayConfiguration
from parallax.replay.timing import ReplayScheduleBuilder, ReplayScheduleEntry


@dataclass(frozen=True, slots=True)
class ReplayPacketEntry:
    """One payload-free packet paired with its deterministic replay timing."""

    schedule: ReplayScheduleEntry
    packet: PacketMetadata

    @property
    def scheduled_offset_seconds(self) -> float:
        """Return the embedded schedule's absolute replay offset."""
        return self.schedule.scheduled_offset_seconds


def iter_pcap_replay_entries(
    source: str | Path,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
    configuration: ReplayConfiguration | None = None,
) -> Iterator[ReplayPacketEntry]:
    """Yield packet metadata paired with deterministic replay schedule entries."""
    builder = ReplayScheduleBuilder(configuration=configuration)

    for packet in iter_pcap_packet_metadata(source, size_policy=size_policy):
        yield ReplayPacketEntry(
            schedule=builder.schedule(packet.timestamp_seconds),
            packet=packet,
        )
