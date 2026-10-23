from pathlib import Path

import pytest
from PIL import Image

from industrial_vision.data.inspection import (
    discover_image_files,
    inspect_dataset,
)


def _save_rgb(path: Path, size: tuple[int, int] = (10, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(0, 0, 0)).save(path)


def _save_mask(path: Path, *, non_empty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (10, 20), color=0)
    if non_empty:
        image.putpixel((0, 0), 255)
    image.save(path)


def test_inspect_dataset_summarizes_structure(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "train" / "sample.jpg")
    _save_mask(tmp_path / "ground_truth" / "sample.png", non_empty=True)

    inspection = inspect_dataset(tmp_path)

    assert inspection.total_images == 2
    assert inspection.by_top_level == {"ground_truth": 1, "train": 1}
    assert inspection.by_extension == {".jpg": 1, ".png": 1}
    assert inspection.by_size == {"10x20": 2}
    assert inspection.mask_candidates == 1
    assert inspection.non_empty_mask_candidates == 1
    assert inspection.unreadable_files == ()


def test_discover_image_files_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_image_files(tmp_path / "missing")
