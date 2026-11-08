"""Command line: ``uv run python -m alife [flags]`` (SPEC 7; flags are decision M8).

Arguments are checked here, before anything imports Ursina, so a bad flag or an unknown life form
id exits without opening a window.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from alife.lifeforms import registry

EXIT_USAGE = 2  # same code argparse uses for a bad command line


def _positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (math.isfinite(value) and value > 0):  # NaN fails every comparison; check it first
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {text!r}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="alife",
        description="Artificial Life Simulator: pick a life form and watch it simulate in 3D.",
    )
    parser.add_argument(
        "--smoke",
        type=_positive_seconds,
        metavar="SECONDS",
        help="quit after SECONDS, the same way as closing the window (for tests)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        metavar="PATH",
        help="save a PNG of the window after 60 frames, then quit",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--lifeform",
        metavar="ID",
        help="skip the menu and start this life form fresh (asks first over a save)",
    )
    start.add_argument(
        "--continue",
        dest="continue_id",
        metavar="ID",
        help="skip the menu and continue this life form from its save",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="seed for a fresh start (default: a random seed)",
    )
    parser.add_argument(
        "--log-keys",
        action="store_true",
        help="print each key and its modifiers as they arrive (to check key names)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    known = registry()
    for flag, value in (("--lifeform", args.lifeform), ("--continue", args.continue_id)):
        if value is not None and value not in known:
            print(
                f"alife: unknown life form {value!r} for {flag}. Known ids: {', '.join(known)}",
                file=sys.stderr,
            )
            return EXIT_USAGE
    from alife.app import App  # imported here: it pulls in Ursina

    return App(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
