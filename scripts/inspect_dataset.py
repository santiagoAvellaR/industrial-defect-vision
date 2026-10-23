import argparse
from pathlib import Path

from industrial_vision.data.inspection import format_report, inspect_ksdd2


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local KSDD2 dataset.")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    report = inspect_ksdd2(args.data_dir)
    print(format_report(args.data_dir, report))


if __name__ == "__main__":
    main()
