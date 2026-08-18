"""Typed contract for the MIT Lincoln Laboratory VNAT dataset."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath
from typing import Final

RAW_COLUMNS: Final = (
    "connection",
    "timestamps",
    "sizes",
    "directions",
    "file_names",
)
FEATURE_LABEL_COLUMN: Final = "labels"
FEATURE_COUNT: Final = 129
WINDOW_SECONDS: Final = 40.96
TIME_BIN_SECONDS: Final = 0.01
MIN_PACKETS_PER_WINDOW: Final = 20


class VpnStatus(StrEnum):
    """Whether traffic was observed inside or outside the VPN tunnel."""

    VPN = "vpn"
    NON_VPN = "nonvpn"


class Application(StrEnum):
    """Applications represented in VNAT release 1."""

    NETFLIX = "netflix"
    YOUTUBE = "youtube"
    VIMEO = "vimeo"
    VOIP = "voip"
    SKYPE_CHAT = "skype-chat"
    SSH = "ssh"
    RDP = "rdp"
    SFTP = "sftp"
    RSYNC = "rsync"
    SCP = "scp"


class TrafficCategory(StrEnum):
    """Category labels supplied by VNAT release 1."""

    STREAMING = "STREAMING"
    VOIP = "VOIP"
    CHAT = "CHAT"
    C2 = "C2"
    FILE_TRANSFER = "FILE_TRANSFER"


APPLICATION_TO_CATEGORY: Final = {
    Application.NETFLIX: TrafficCategory.STREAMING,
    Application.YOUTUBE: TrafficCategory.STREAMING,
    Application.VIMEO: TrafficCategory.STREAMING,
    Application.VOIP: TrafficCategory.VOIP,
    Application.SKYPE_CHAT: TrafficCategory.CHAT,
    Application.SSH: TrafficCategory.C2,
    Application.RDP: TrafficCategory.C2,
    Application.SFTP: TrafficCategory.FILE_TRANSFER,
    Application.RSYNC: TrafficCategory.FILE_TRANSFER,
    Application.SCP: TrafficCategory.FILE_TRANSFER,
}


@dataclass(frozen=True, slots=True)
class CaptureMetadata:
    """Labels derived from one VNAT capture filename."""

    capture_id: str
    vpn_status: VpnStatus
    application: Application
    category: TrafficCategory


class VnatFilenameError(ValueError):
    """Raised when a VNAT capture filename violates the release contract."""


def parse_capture_filename(file_name: str | PurePath) -> CaptureMetadata:
    """Parse authoritative labels and a split-group ID from a VNAT filename."""
    capture_id = PurePath(file_name).name
    normalized = capture_id.casefold()

    if not normalized.endswith(".pcap"):
        raise VnatFilenameError(f"expected a .pcap filename, got {capture_id!r}")

    if normalized.startswith("nonvpn_"):
        vpn_status = VpnStatus.NON_VPN
    elif normalized.startswith("vpn_"):
        vpn_status = VpnStatus.VPN
    else:
        raise VnatFilenameError(f"expected a vpn_ or nonvpn_ prefix, got {capture_id!r}")

    applications = [application for application in Application if application.value in normalized]
    if len(applications) != 1:
        raise VnatFilenameError(
            "expected exactly one application keyword in "
            f"{capture_id!r}, found {[item.value for item in applications]}"
        )

    application = applications[0]
    return CaptureMetadata(
        capture_id=capture_id,
        vpn_status=vpn_status,
        application=application,
        category=APPLICATION_TO_CATEGORY[application],
    )
