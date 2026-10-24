import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_images(
    dataset_root: Path,
    manifests_dir: Path,
) -> list[list[tuple[str, str]]]:
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        manifest = Path(manifests_dir) / f"{split}.csv"
        with manifest.open(newline="") as file:
            for row in csv.DictReader(file):
                relative_path = row["image"]
                image_path = Path(dataset_root) / relative_path
                by_hash[_file_hash(image_path)].append((split, relative_path))

    return [group for group in by_hash.values() if len(group) > 1]


def cross_split_duplicates(
    groups: list[list[tuple[str, str]]],
) -> list[list[tuple[str, str]]]:
    return [
        group
        for group in groups
        if len({split for split, _ in group}) > 1
    ]
