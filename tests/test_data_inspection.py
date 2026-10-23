from pathlib import Path

from PIL import Image

from industrial_vision.data.inspection import find_pairs, inspect_split


def save_pair(root: Path, split: str, name: str, *, defective: bool = False) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (10, 20)).save(split_dir / f"{name}.png")

    mask = Image.new("L", (10, 20))
    if defective:
        mask.putpixel((0, 0), 255)
    mask.save(split_dir / f"{name}_GT.png")


def test_inspect_split_counts_defective_and_clean_samples(tmp_path: Path) -> None:
    save_pair(tmp_path, "train", "10000")
    save_pair(tmp_path, "train", "10001", defective=True)

    stats = inspect_split(tmp_path, "train")

    assert stats["samples"] == 2
    assert stats["defective"] == 1
    assert stats["clean"] == 1
    assert stats["missing_masks"] == []
    assert stats["orphan_masks"] == []
    assert stats["size_mismatches"] == []


def test_find_pairs_reports_missing_and_orphan_masks(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    Image.new("RGB", (10, 20)).save(split_dir / "10000.png")
    Image.new("L", (10, 20)).save(split_dir / "99999_GT.png")

    pairs, missing_masks, orphan_masks = find_pairs(tmp_path, "train")

    assert pairs == []
    assert missing_masks == ["10000.png"]
    assert orphan_masks == ["99999_GT.png"]
