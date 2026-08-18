"""Command-line entry point for Parallax."""

from parallax import __version__


def main() -> None:
    """Print the installed Parallax version."""
    print(f"Parallax {__version__}")
