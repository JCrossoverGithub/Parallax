"""Tests for the initial Parallax command-line entry point."""

from pytest import CaptureFixture

from parallax import __version__
from parallax.cli import main


def test_package_version_matches_initial_release() -> None:
    assert __version__ == "0.1.0"


def test_main_reports_name_and_version(capsys: CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out == "Parallax 0.1.0\n"
