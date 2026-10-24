"""Check exact image duplicates across KSDD2 splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_vision.data.duplicates import (
    cross_split_duplicates,
    find_duplicate_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check exact duplicate images across train, val and test."
    )
    parser.add_argument("data_dir", type=Path, help="KSDD2 dataset root.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train.csv, val.csv and test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = find_duplicate_images(args.data_dir, args.splits_dir)
    leakage = cross_split_duplicates(groups)

    print(f"duplicate groups: {len(groups)}")
    print(f"cross-split duplicate groups: {len(leakage)}")

    for group in leakage:
        print("\ncross-split duplicate:")
        for split, path in group:
            print(f"  {split}: {path}")

    if leakage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
