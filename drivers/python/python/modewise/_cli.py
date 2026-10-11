import sys

from ._core import cli


def main() -> None:
    """Entry point of the `modewise` command installed with the package."""
    sys.exit(cli(["modewise", *sys.argv[1:]]))
