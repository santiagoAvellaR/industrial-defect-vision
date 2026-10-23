import argparse
from pathlib import Path

from industrial_vision.data.splits import create_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/validation/test manifests for KSDD2."
    )
    parser.add_argument("data_dir", type=Path, help="Root directory of KSDD2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = create_manifests(
        args.data_dir,
        Path("data/splits"),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    for name, samples in splits.items():
        defective = sum(label for _, _, label in samples)
        print(
            f"{name}: {len(samples)} samples "
            f"({defective} defective, {len(samples) - defective} clean)"
        )


if __name__ == "__main__":
    main()
