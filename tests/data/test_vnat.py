from pathlib import PurePath

import pytest

from parallax.data.vnat import (
    APPLICATION_TO_CATEGORY,
    FEATURE_COUNT,
    FEATURE_LABEL_COLUMN,
    MIN_PACKETS_PER_WINDOW,
    RAW_COLUMNS,
    TIME_BIN_SECONDS,
    WINDOW_SECONDS,
    Application,
    TrafficCategory,
    VnatFilenameError,
    VpnStatus,
    parse_capture_filename,
)


@pytest.mark.parametrize(
    ("keyword", "application", "category"),
    [
        ("netflix", Application.NETFLIX, TrafficCategory.STREAMING),
        ("youtube", Application.YOUTUBE, TrafficCategory.STREAMING),
        ("vimeo", Application.VIMEO, TrafficCategory.STREAMING),
        ("voip", Application.VOIP, TrafficCategory.VOIP),
        ("skype-chat", Application.SKYPE_CHAT, TrafficCategory.CHAT),
        ("ssh", Application.SSH, TrafficCategory.C2),
        ("rdp", Application.RDP, TrafficCategory.C2),
        ("sftp", Application.SFTP, TrafficCategory.FILE_TRANSFER),
        ("rsync", Application.RSYNC, TrafficCategory.FILE_TRANSFER),
        ("scp", Application.SCP, TrafficCategory.FILE_TRANSFER),
    ],
)
@pytest.mark.parametrize(
    ("prefix", "vpn_status"),
    [("vpn", VpnStatus.VPN), ("nonvpn", VpnStatus.NON_VPN)],
)
def test_parse_capture_filename(
    keyword: str,
    application: Application,
    category: TrafficCategory,
    prefix: str,
    vpn_status: VpnStatus,
) -> None:
    file_name = f"{prefix}_{keyword}_capture1.pcap"

    metadata = parse_capture_filename(file_name)

    assert metadata.capture_id == file_name
    assert metadata.vpn_status is vpn_status
    assert metadata.application is application
    assert metadata.category is category
    assert APPLICATION_TO_CATEGORY[application] is category


def test_parse_capture_filename_accepts_paths_and_case() -> None:
    metadata = parse_capture_filename(PurePath("release", "VPN_YOUTUBE_CAPTURE2.PCAP"))

    assert metadata.capture_id == "VPN_YOUTUBE_CAPTURE2.PCAP"
    assert metadata.vpn_status is VpnStatus.VPN
    assert metadata.application is Application.YOUTUBE


@pytest.mark.parametrize(
    ("file_name", "message"),
    [
        ("vpn_youtube_capture1.txt", "expected a .pcap filename"),
        ("proxy_youtube_capture1.pcap", "expected a vpn_ or nonvpn_ prefix"),
        ("vpn_unknown_capture1.pcap", "expected exactly one application keyword"),
        (
            "vpn_youtube_netflix_capture1.pcap",
            "expected exactly one application keyword",
        ),
    ],
)
def test_parse_capture_filename_rejects_invalid_names(file_name: str, message: str) -> None:
    with pytest.raises(VnatFilenameError, match=message):
        parse_capture_filename(file_name)


def test_release_contract_constants() -> None:
    assert RAW_COLUMNS == (
        "connection",
        "timestamps",
        "sizes",
        "directions",
        "file_names",
    )
    assert FEATURE_LABEL_COLUMN == "labels"
    assert FEATURE_COUNT == 129
    assert WINDOW_SECONDS == 40.96
    assert TIME_BIN_SECONDS == 0.01
    assert MIN_PACKETS_PER_WINDOW == 20
