"""Command-line entry point for MatrixMan checks."""

from __future__ import annotations

import argparse

from .compatibility import main as compatibility_main


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan utilities")
    parser.add_argument("--check", action="store_true", help="run the OpenGL compatibility probe")
    args = parser.parse_args()
    if args.check:
        return compatibility_main()
    parser.error("use --check")


if __name__ == "__main__":
    raise SystemExit(main())
