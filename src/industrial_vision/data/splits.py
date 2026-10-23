import csv
import random
from pathlib import Path

from PIL import Image

from industrial_vision.data.inspection import find_pairs


def load_samples(root: Path, split: str) -> list[tuple[Path, Path, int]]:
    pairs, missing_masks, orphan_masks = find_pairs(root, split)

    if missing_masks or orphan_masks:
        raise ValueError(
            f"Invalid {split} split: {len(missing_masks)} missing masks and "
            f"{len(orphan_masks)} orphan masks"
        )

    samples = []
    for image_path, mask_path in pairs:
        with Image.open(mask_path) as mask:
            label = int(mask.getbbox() is not None)
        samples.append((image_path, mask_path, label))

    return samples


def stratified_split(
    samples: list[tuple[Path, Path, int]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path, int]], list[tuple[Path, Path, int]]]:
    positives = [sample for sample in samples if sample[2] == 1]
    negatives = [sample for sample in samples if sample[2] == 0]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_val_positive = round(len(positives) * val_fraction)
    n_val_negative = round(len(negatives) * val_fraction)

    val_samples = positives[:n_val_positive] + negatives[:n_val_negative]
    train_samples = positives[n_val_positive:] + negatives[n_val_negative:]

    train_samples.sort(key=lambda sample: sample[0].name)
    val_samples.sort(key=lambda sample: sample[0].name)
    return train_samples, val_samples


def write_manifest(
    path: Path,
    samples: list[tuple[Path, Path, int]],
    dataset_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "mask", "label"])
        for image_path, mask_path, label in samples:
            writer.writerow([
                image_path.relative_to(dataset_root),
                mask_path.relative_to(dataset_root),
                label,
            ])


def create_manifests(
    dataset_root: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path, int]]]:
    dataset_root = Path(dataset_root)
    train_samples = load_samples(dataset_root, "train")
    test_samples = load_samples(dataset_root, "test")

    train_samples, val_samples = stratified_split(
        train_samples,
        val_fraction=val_fraction,
        seed=seed,
    )

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for name, samples in splits.items():
        write_manifest(output_dir / f"{name}.csv", samples, dataset_root)

    return splits
