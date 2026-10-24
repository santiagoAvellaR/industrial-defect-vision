import csv
from pathlib import Path

from industrial_vision.data.duplicates import (
    cross_split_duplicates,
    find_duplicate_images,
)


def _write_manifest(path: Path, image: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "mask", "label"])
        writer.writerow([image, "unused.png", 0])


def test_detects_duplicate_across_splits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    manifests_dir = tmp_path / "splits"
    dataset_root.mkdir()

    (dataset_root / "train.png").write_bytes(b"same-image")
    (dataset_root / "val.png").write_bytes(b"same-image")
    (dataset_root / "test.png").write_bytes(b"different-image")

    _write_manifest(manifests_dir / "train.csv", "train.png")
    _write_manifest(manifests_dir / "val.csv", "val.png")
    _write_manifest(manifests_dir / "test.csv", "test.png")

    groups = find_duplicate_images(dataset_root, manifests_dir)
    leakage = cross_split_duplicates(groups)

    assert leakage == [[("train", "train.png"), ("val", "val.png")]]


def test_no_duplicates_returns_empty_lists(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    manifests_dir = tmp_path / "splits"
    dataset_root.mkdir()

    for split in ("train", "val", "test"):
        image = f"{split}.png"
        (dataset_root / image).write_bytes(split.encode())
        _write_manifest(manifests_dir / f"{split}.csv", image)

    groups = find_duplicate_images(dataset_root, manifests_dir)

    assert groups == []
    assert cross_split_duplicates(groups) == []
