import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.evaluation.classification import collect_classification_outputs
from industrial_vision.models.baseline_cnn import BaselineCNN
from industrial_vision.training.classification import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation errors for the CNN classification baseline."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/splits/val.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/error_analysis"),
    )
    return parser.parse_args()


def error_type(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 0:
        return "tn"
    if label == 0 and prediction == 1:
        return "fp"
    return "fn"


def save_score_distribution(records: list[dict], threshold: float, path: Path) -> None:
    clean_scores = [row["score"] for row in records if row["label"] == 0]
    defect_scores = [row["score"] for row in records if row["label"] == 1]

    figure, axis = plt.subplots()
    axis.hist(clean_scores, bins=20, range=(0, 1), histtype="step", label="Clean")
    axis.hist(defect_scores, bins=20, range=(0, 1), histtype="step", label="Defective")
    axis.axvline(threshold, linestyle="--", label=f"Threshold {threshold:.2f}")
    axis.set_xlabel("Defect score")
    axis.set_ylabel("Number of images")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_defect_area_comparison(records: list[dict], path: Path) -> None:
    tp_areas = [row["mask_area_ratio"] * 100 for row in records if row["error"] == "tp"]
    fn_areas = [row["mask_area_ratio"] * 100 for row in records if row["error"] == "fn"]

    figure, axis = plt.subplots()
    axis.boxplot([tp_areas, fn_areas], labels=["TP", "FN"])
    axis.set_ylabel("Defect area (% of image)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_error_examples(
    records: list[dict],
    dataset_root: Path,
    error: str,
    path: Path,
    limit: int = 12,
) -> None:
    selected = [row for row in records if row["error"] == error]
    reverse = error == "fp"
    selected = sorted(selected, key=lambda row: row["score"], reverse=reverse)[:limit]

    if not selected:
        return

    columns = 4
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, row in zip(axes, selected):
        image_path = dataset_root / row["image"]
        mask_path = dataset_root / row["mask"]

        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        if row["label"] == 1:
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(mask.convert("L")) > 0
            axis.contour(mask_array, levels=[0.5], linewidths=1)

        axis.set_title(
            f"{error.upper()} score={row['score']:.2f}\n{Path(row['image']).name}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes[len(selected):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    global_pool = checkpoint.get("global_pool", "avg")
    model = BaselineCNN(global_pool=global_pool).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = KSDD2Dataset(args.dataset_root, args.val_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probabilities, labels, mask_area_ratios = collect_classification_outputs(
        model, loader, device
    )
    metrics = classification_metrics(probabilities, labels, args.threshold)
    predictions = (probabilities >= args.threshold).long()

    records = []
    for sample, score, label, prediction, mask_area_ratio in zip(
        dataset.samples,
        probabilities.tolist(),
        labels.tolist(),
        predictions.tolist(),
        mask_area_ratios.tolist(),
    ):
        records.append(
            {
                "image": sample["image"],
                "mask": sample["mask"],
                "label": label,
                "score": score,
                "prediction": prediction,
                "error": error_type(label, prediction),
                "mask_area_ratio": mask_area_ratio,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "validation_predictions.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    save_score_distribution(
        records,
        args.threshold,
        args.output_dir / "score_distribution.png",
    )
    save_defect_area_comparison(
        records,
        args.output_dir / "defect_area_tp_vs_fn.png",
    )
    save_error_examples(
        records,
        args.dataset_root,
        "fn",
        args.output_dir / "false_negatives.png",
    )
    save_error_examples(
        records,
        args.dataset_root,
        "fp",
        args.output_dir / "false_positives.png",
    )

    tp_areas = [row["mask_area_ratio"] for row in records if row["error"] == "tp"]
    fn_areas = [row["mask_area_ratio"] for row in records if row["error"] == "fn"]

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"global_pool: {global_pool}")
    print(f"threshold: {args.threshold:.2f}")
    print(
        f"tp={metrics['tp']} tn={metrics['tn']} "
        f"fp={metrics['fp']} fn={metrics['fn']}"
    )
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )

    if tp_areas and fn_areas:
        print(
            "median defect area: "
            f"TP={np.median(tp_areas) * 100:.3f}% "
            f"FN={np.median(fn_areas) * 100:.3f}%"
        )

    print(f"saved predictions: {csv_path}")
    print(f"saved figures: {args.output_dir}")


if __name__ == "__main__":
    main()
