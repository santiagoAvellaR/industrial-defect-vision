import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset


def _save_sample(root: Path, name: str, defective: bool) -> tuple[str, str, int]:
    image_path = root / "train" / f"{name}.png"
    mask_path = root / "train" / f"{name}_GT.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (10, 20), color=(20, 40, 60)).save(image_path)
    mask = Image.new("L", (10, 20), color=0)
    if defective:
        mask.putpixel((5, 10), 255)
    mask.save(mask_path)

    return (
        str(image_path.relative_to(root)),
        str(mask_path.relative_to(root)),
        int(defective),
    )


def _write_manifest(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "mask", "label"])
        writer.writerows(rows)


def test_dataset_loads_image_mask_and_label(tmp_path: Path) -> None:
    row = _save_sample(tmp_path, "10000", defective=True)
    manifest = tmp_path / "train.csv"
    _write_manifest(manifest, [row])

    dataset = KSDD2Dataset(tmp_path, manifest, image_size=(8, 4))
    image, mask, label = dataset[0]

    assert image.shape == (3, 8, 4)
    assert mask.shape == (1, 8, 4)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert label == 1


def test_dataset_can_be_batched(tmp_path: Path) -> None:
    rows = [
        _save_sample(tmp_path, "10000", defective=True),
        _save_sample(tmp_path, "10001", defective=False),
    ]
    manifest = tmp_path / "train.csv"
    _write_manifest(manifest, rows)

    dataset = KSDD2Dataset(tmp_path, manifest, image_size=(8, 4))
    images, masks, labels = next(iter(DataLoader(dataset, batch_size=2)))

    assert images.shape == (2, 3, 8, 4)
    assert masks.shape == (2, 1, 8, 4)
    assert labels.tolist() == [1, 0]
