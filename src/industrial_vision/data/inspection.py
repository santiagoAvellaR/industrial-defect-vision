"""Utilities for inspecting image datasets before model development."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_HINTS = {
    "annotation",
    "annotations",
    "ground_truth",
    "groundtruth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
}


@dataclass(frozen=True)
class DatasetInspection:
    """Summary produced from a recursive scan of an image dataset."""

    root: Path
    total_images: int
    by_top_level: dict[str, int]
    by_extension: dict[str, int]
    by_size: dict[str, int]
    by_mode: dict[str, int]
    mask_candidates: int
    non_empty_mask_candidates: int
    unreadable_files: tuple[Path, ...]
    samples_by_top_level: dict[str, tuple[Path, ...]]


def discover_image_files(root: Path) -> list[Path]:
    """Return supported image files below ``root`` in deterministic order."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _top_level_group(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "."


def _looks_like_mask(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    directory_parts = {part.lower() for part in relative.parts[:-1]}
    stem_tokens = set(relative.stem.lower().replace("-", "_").split("_"))
    return bool((directory_parts | stem_tokens) & MASK_HINTS)


def inspect_dataset(root: Path, *, samples_per_group: int = 3) -> DatasetInspection:
    """Inspect dataset structure and lightweight image metadata.

    The function intentionally makes no assumptions about dataset-specific
    image/mask pairing. Mask statistics are only heuristic candidates based on
    common directory or filename hints.
    """

    root = Path(root).resolve()
    image_files = discover_image_files(root)

    by_top_level: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    by_size: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    samples: defaultdict[str, list[Path]] = defaultdict(list)
    unreadable: list[Path] = []
    mask_candidates = 0
    non_empty_mask_candidates = 0

    for path in image_files:
        group = _top_level_group(root, path)
        by_top_level[group] += 1
        by_extension[path.suffix.lower()] += 1
        if len(samples[group]) < samples_per_group:
            samples[group].append(path.relative_to(root))

        is_mask_candidate = _looks_like_mask(root, path)
        if is_mask_candidate:
            mask_candidates += 1

        try:
            with Image.open(path) as image:
                width, height = image.size
                by_size[f"{width}x{height}"] += 1
                by_mode[image.mode] += 1

                if is_mask_candidate:
                    extrema = image.convert("L").getextrema()
                    if extrema is not None and extrema[1] > 0:
                        non_empty_mask_candidates += 1
        except (OSError, ValueError):
            unreadable.append(path.relative_to(root))

    return DatasetInspection(
        root=root,
        total_images=len(image_files),
        by_top_level=dict(sorted(by_top_level.items())),
        by_extension=dict(sorted(by_extension.items())),
        by_size=dict(by_size.most_common()),
        by_mode=dict(sorted(by_mode.items())),
        mask_candidates=mask_candidates,
        non_empty_mask_candidates=non_empty_mask_candidates,
        unreadable_files=tuple(unreadable),
        samples_by_top_level={
            group: tuple(paths) for group, paths in sorted(samples.items())
        },
    )


def _format_counts(title: str, values: dict[str, int]) -> Iterable[str]:
    yield f"{title}:"
    if not values:
        yield "  (none)"
        return
    for key, value in values.items():
        yield f"  {key}: {value}"


def format_inspection(inspection: DatasetInspection) -> str:
    """Format an inspection summary for terminal output."""

    lines = [
        f"Dataset root: {inspection.root}",
        f"Image files: {inspection.total_images}",
        "",
    ]
    lines.extend(_format_counts("Top-level groups", inspection.by_top_level))
    lines.append("")
    lines.extend(_format_counts("Extensions", inspection.by_extension))
    lines.append("")
    lines.extend(_format_counts("Image sizes", inspection.by_size))
    lines.append("")
    lines.extend(_format_counts("Image modes", inspection.by_mode))
    lines.extend(
        [
            "",
            f"Mask candidates (heuristic): {inspection.mask_candidates}",
            "Non-empty mask candidates (heuristic): "
            f"{inspection.non_empty_mask_candidates}",
            f"Unreadable image files: {len(inspection.unreadable_files)}",
            "",
            "Sample paths:",
        ]
    )

    for group, paths in inspection.samples_by_top_level.items():
        lines.append(f"  {group}:")
        lines.extend(f"    {path}" for path in paths)

    if inspection.unreadable_files:
        lines.extend(["", "Unreadable files:"])
        lines.extend(f"  {path}" for path in inspection.unreadable_files)

    return "\n".join(lines)
