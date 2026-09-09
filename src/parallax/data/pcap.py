"""Strict metadata-only parsing for classic Raw-IP PCAP captures."""

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import dpkt  # type: ignore[import-untyped]

from parallax.data.packets import (
    IP_PROTOCOL_ICMP as IP_PROTOCOL_ICMP,
)
from parallax.data.packets import (
    IP_PROTOCOL_TCP as IP_PROTOCOL_TCP,
)
from parallax.data.packets import (
    IP_PROTOCOL_UDP as IP_PROTOCOL_UDP,
)
from parallax.data.packets import (
    PacketDecodeError,
    decode_raw_ipv4_packet,
)
from parallax.data.packets import (
    PacketMetadata as PacketMetadata,
)
from parallax.data.packets import (
    PacketSizePolicy as PacketSizePolicy,
)

PCAP_LINKTYPE_RAW_IP: Final = 101


class PcapReadError(ValueError):
    """Raised when a PCAP cannot be converted to supported packet metadata."""


def iter_pcap_packet_metadata(
    source: str | Path,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
) -> Iterator[PacketMetadata]:
    """Yield supported packet metadata from one timestamp-ordered classic PCAP."""
    if not isinstance(size_policy, PacketSizePolicy):
        raise TypeError("size_policy must be a PacketSizePolicy")

    source_path = Path(source)
    if not source_path.is_file():
        raise PcapReadError(f"PCAP file does not exist: {source_path}")

    with source_path.open("rb") as stream:
        try:
            reader = dpkt.pcap.Reader(stream)
        except (ValueError, dpkt.dpkt.Error) as error:
            raise PcapReadError(f"invalid classic PCAP header: {source_path}") from error

        linktype = int(reader.datalink())
        if linktype != PCAP_LINKTYPE_RAW_IP:
            raise PcapReadError(
                f"unsupported PCAP link type {linktype}; expected Raw IP ({PCAP_LINKTYPE_RAW_IP})"
            )

        previous_timestamp: float | None = None
        packets_read = 0

        try:
            for packet_number, (raw_timestamp, raw_packet) in enumerate(
                reader,
                start=1,
            ):
                timestamp = float(raw_timestamp)

                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise PcapReadError(
                        f"packet {packet_number}: timestamp precedes the previous packet"
                    )

                previous_timestamp = timestamp

                try:
                    yield decode_raw_ipv4_packet(
                        timestamp,
                        bytes(raw_packet),
                        size_policy=size_policy,
                    )
                except PacketDecodeError as error:
                    raise PcapReadError(f"packet {packet_number}: {error}") from error

                packets_read = packet_number

        except dpkt.dpkt.Error as error:
            raise PcapReadError(
                f"could not read PCAP record after packet {packets_read}"
            ) from error
