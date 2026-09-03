"""Runtime feature construction from labeled VNAT PCAP captures."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from parallax.data.pcap import PacketSizePolicy
from parallax.data.pcap_windowing import extract_vnat_pcap_windows
from parallax.data.vnat import CaptureMetadata
from parallax.data.windowing import WindowExtractionConfig
from parallax.features.calculator import (
    FeatureCalculationConfig,
    calculate_feature_vector,
)
from parallax.features.schema import Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class VnatWindowFeature:
    """Traceable feature vector without retained raw packet sequences."""

    window_id: str
    capture: CaptureMetadata
    flow_id: str
    window_index: int
    start_offset_seconds: float
    end_offset_seconds: float
    packet_count: int
    values: Float32Array


def extract_vnat_pcap_features(
    source: str | Path,
    *,
    size_policy: PacketSizePolicy = PacketSizePolicy.RELEASE_COMPATIBLE,
    window_config: WindowExtractionConfig | None = None,
    feature_config: FeatureCalculationConfig | None = None,
) -> Iterator[VnatWindowFeature]:
    """Yield shared-pipeline features for eligible windows in one VNAT PCAP."""
    for window in extract_vnat_pcap_windows(
        source,
        size_policy=size_policy,
        config=window_config,
    ):
        yield VnatWindowFeature(
            window_id=window.window_id,
            capture=window.capture,
            flow_id=window.flow_id,
            window_index=window.window_index,
            start_offset_seconds=window.start_offset_seconds,
            end_offset_seconds=window.end_offset_seconds,
            packet_count=window.packet_count,
            values=calculate_feature_vector(
                window.timestamps,
                window.sizes,
                window.directions,
                config=feature_config,
            ),
        )
