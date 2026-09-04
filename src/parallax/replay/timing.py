"""Deterministic timestamp scheduling for replay sources."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite

from parallax.replay.domain import ReplayConfiguration, ReplayDomainError


class ReplayTimingError(ReplayDomainError):
    """Raised when source timestamps cannot form a deterministic replay schedule."""


@dataclass(frozen=True, slots=True)
class ReplayScheduleEntry:
    """Timing target for one source packet relative to replay start."""

    packet_number: int
    source_timestamp_seconds: float
    source_offset_seconds: float
    scheduled_offset_seconds: float


def iter_replay_schedule(
    timestamps: Iterable[float],
    *,
    configuration: ReplayConfiguration | None = None,
) -> Iterator[ReplayScheduleEntry]:
    """Yield deterministic absolute replay offsets for ordered source timestamps."""
    config = configuration if configuration is not None else ReplayConfiguration()

    origin_timestamp: float | None = None
    previous_timestamp: float | None = None

    for packet_number, timestamp in enumerate(timestamps, start=1):
        if not isfinite(timestamp):
            raise ReplayTimingError(f"packet {packet_number}: source timestamp must be finite")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ReplayTimingError(
                f"packet {packet_number}: source timestamp precedes the previous packet"
            )

        if origin_timestamp is None:
            origin_timestamp = timestamp

        source_offset = timestamp - origin_timestamp
        time_scale = config.time_scale
        scheduled_offset = 0.0 if time_scale is None else source_offset / time_scale

        yield ReplayScheduleEntry(
            packet_number=packet_number,
            source_timestamp_seconds=timestamp,
            source_offset_seconds=source_offset,
            scheduled_offset_seconds=scheduled_offset,
        )

        previous_timestamp = timestamp
