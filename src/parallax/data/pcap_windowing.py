"""VNAT PCAP adapter for the shared observation-window implementation."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from parallax.data.flows import BidirectionalFlow, group_bidirectional_flows
from parallax.data.pcap import (
    PacketSizePolicy,
    iter_pcap_packet_metadata,
)
from parallax.data.vnat import RAW_COLUMNS
from parallax.data.windowing import (
    ObservationWindow,
    WindowExtractionConfig,
    extract_capture_windows,
)


def extract_vnat_pcap_windows(
    source: str | Path,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
    config: WindowExtractionConfig | None = None,
) -> Iterator[ObservationWindow]:
    """Parse one labeled VNAT PCAP through the shared windowing path."""
    source_path = Path(source)
    flows = group_bidirectional_flows(
        iter_pcap_packet_metadata(source_path, size_policy=size_policy)
    )
    frame = _flow_frame(flows, source_path.name)
    return extract_capture_windows(frame, config=config)


def _flow_frame(flows: tuple[BidirectionalFlow, ...], capture_id: str) -> pd.DataFrame:
    records = [
        {
            "connection": flow.connection,
            "timestamps": list(flow.timestamps),
            "sizes": list(flow.sizes),
            "directions": list(flow.directions),
            "file_names": capture_id,
        }
        for flow in flows
    ]
    return pd.DataFrame.from_records(records, columns=RAW_COLUMNS)
