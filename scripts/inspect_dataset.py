"""CLI entry point for lightweight dataset inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_vision.data.inspection import format_inspection, inspect_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect image dataset structure and basic metadata."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root directory of the extracted dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_dataset(args.data_dir)
    print(format_inspection(inspection))


if __name__ == "__main__":
    main()
