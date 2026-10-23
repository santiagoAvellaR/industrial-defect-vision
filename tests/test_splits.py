from pathlib import Path

from PIL import Image

from industrial_vision.data.splits import create_manifests, load_samples, stratified_split


def save_sample(folder: Path, name: str, defective: bool) -> None:
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    mask = Image.new("L", (8, 8), color=0)
    if defective:
        mask.putpixel((0, 0), 255)

    image.save(folder / f"{name}.png")
    mask.save(folder / f"{name}_GT.png")


def test_stratified_split_preserves_class_balance(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    for index in range(10):
        save_sample(train_dir, f"positive_{index}", defective=True)
    for index in range(20):
        save_sample(train_dir, f"negative_{index}", defective=False)

    samples = load_samples(tmp_path, "train")
    train, val = stratified_split(samples, val_fraction=0.2, seed=42)

    assert len(train) == 24
    assert len(val) == 6
    assert sum(label for _, _, label in val) == 2


def test_create_manifests_is_reproducible(tmp_path: Path) -> None:
    for split in ("train", "test"):
        folder = tmp_path / split
        folder.mkdir()
        for index in range(5):
            save_sample(folder, f"{split}_{index}", defective=index == 0)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    create_manifests(tmp_path, first_dir, seed=7)
    create_manifests(tmp_path, second_dir, seed=7)

    for name in ("train", "val", "test"):
        assert (first_dir / f"{name}.csv").read_text() == (
            second_dir / f"{name}.csv"
        ).read_text()
