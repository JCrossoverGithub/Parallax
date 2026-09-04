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


class ReplayScheduleBuilder:
    """Incrementally build one deterministic replay schedule."""

    __slots__ = (
        "_configuration",
        "_origin_timestamp",
        "_packet_number",
        "_previous_timestamp",
    )

    def __init__(
        self,
        *,
        configuration: ReplayConfiguration | None = None,
    ) -> None:
        self._configuration = configuration if configuration is not None else ReplayConfiguration()
        self._origin_timestamp: float | None = None
        self._previous_timestamp: float | None = None
        self._packet_number = 0

    def schedule(self, timestamp: float) -> ReplayScheduleEntry:
        """Schedule the next ordered source timestamp."""
        packet_number = self._packet_number + 1

        if not isfinite(timestamp):
            raise ReplayTimingError(f"packet {packet_number}: source timestamp must be finite")
        if self._previous_timestamp is not None and timestamp < self._previous_timestamp:
            raise ReplayTimingError(
                f"packet {packet_number}: source timestamp precedes the previous packet"
            )

        origin_timestamp = self._origin_timestamp
        if origin_timestamp is None:
            origin_timestamp = timestamp

        source_offset = timestamp - origin_timestamp
        time_scale = self._configuration.time_scale
        scheduled_offset = 0.0 if time_scale is None else source_offset / time_scale

        entry = ReplayScheduleEntry(
            packet_number=packet_number,
            source_timestamp_seconds=timestamp,
            source_offset_seconds=source_offset,
            scheduled_offset_seconds=scheduled_offset,
        )

        self._origin_timestamp = origin_timestamp
        self._previous_timestamp = timestamp
        self._packet_number = packet_number

        return entry


def iter_replay_schedule(
    timestamps: Iterable[float],
    *,
    configuration: ReplayConfiguration | None = None,
) -> Iterator[ReplayScheduleEntry]:
    """Yield deterministic absolute replay offsets for ordered source timestamps."""
    builder = ReplayScheduleBuilder(configuration=configuration)

    for timestamp in timestamps:
        yield builder.schedule(timestamp)
