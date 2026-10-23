from pathlib import Path

from PIL import Image


def find_pairs(root: Path, split: str):
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing KSDD2 split: {split_dir}")

    images = sorted(
        path for path in split_dir.glob("*.png")
        if not path.stem.endswith("_GT")
    )
    masks = {
        path.stem.removesuffix("_GT"): path
        for path in split_dir.glob("*_GT.png")
    }

    pairs = []
    missing_masks = []

    for image_path in images:
        mask_path = masks.pop(image_path.stem, None)
        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))

    orphan_masks = sorted(path.name for path in masks.values())
    return pairs, missing_masks, orphan_masks


def inspect_split(root: Path, split: str) -> dict:
    pairs, missing_masks, orphan_masks = find_pairs(root, split)

    defective = 0
    size_mismatches = []

    for image_path, mask_path in pairs:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                size_mismatches.append(image_path.name)
            if mask.getbbox() is not None:
                defective += 1

    return {
        "samples": len(pairs),
        "defective": defective,
        "clean": len(pairs) - defective,
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "size_mismatches": size_mismatches,
    }


def inspect_ksdd2(root: Path) -> dict[str, dict]:
    root = Path(root)
    return {
        "train": inspect_split(root, "train"),
        "test": inspect_split(root, "test"),
    }


def format_report(root: Path, report: dict[str, dict]) -> str:
    lines = [f"KSDD2 root: {Path(root).resolve()}"]

    for split in ("train", "test"):
        stats = report[split]
        lines.extend([
            "",
            f"{split}:",
            f"  samples: {stats['samples']}",
            f"  defective: {stats['defective']}",
            f"  clean: {stats['clean']}",
            f"  missing masks: {len(stats['missing_masks'])}",
            f"  orphan masks: {len(stats['orphan_masks'])}",
            f"  image/mask size mismatches: {len(stats['size_mismatches'])}",
        ])

        for label, key in (
            ("missing mask", "missing_masks"),
            ("orphan mask", "orphan_masks"),
            ("size mismatch", "size_mismatches"),
        ):
            for name in stats[key]:
                lines.append(f"    {label}: {name}")

    total_samples = sum(report[split]["samples"] for split in report)
    total_defective = sum(report[split]["defective"] for split in report)

    lines.extend([
        "",
        "total:",
        f"  samples: {total_samples}",
        f"  defective: {total_defective}",
        f"  clean: {total_samples - total_defective}",
    ])

    return "\n".join(lines)
