import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.loading import build_model_from_checkpoint
from industrial_vision.training.segmentation import segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation thresholds on the validation split."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/splits/val.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("artifacts/figures/segmentation_threshold_sweep.png"),
    )
    return parser.parse_args()


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = KSDD2Dataset(args.dataset_root, args.val_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    thresholds = torch.linspace(0.10, 0.90, 17).tolist()
    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in thresholds
    }

    with torch.inference_mode():
        for images, masks, _ in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).cpu()

            for threshold in thresholds:
                metrics = segmentation_metrics(probabilities, masks, threshold)
                counts[threshold]["tp"] += int(metrics["tp"])
                counts[threshold]["fp"] += int(metrics["fp"])
                counts[threshold]["fn"] += int(metrics["fn"])

    rows = []

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"checkpoint val_loss: {checkpoint['val_loss']:.4f}")
    print(f"model: {checkpoint.get('model_name', 'unknown')}")
    print()
    print("threshold    dice     iou precision  recall       tp       fp       fn")
    print("--------- ------- ------- --------- ------- -------- -------- --------")

    for threshold in thresholds:
        threshold_counts = counts[threshold]
        metrics = metrics_from_counts(
            threshold_counts["tp"],
            threshold_counts["fp"],
            threshold_counts["fn"],
        )
        rows.append((threshold, metrics))
        print(
            f"{threshold:9.2f} "
            f"{metrics['dice']:7.4f} "
            f"{metrics['iou']:7.4f} "
            f"{metrics['precision']:9.4f} "
            f"{metrics['recall']:7.4f} "
            f"{metrics['tp']:8d} "
            f"{metrics['fp']:8d} "
            f"{metrics['fn']:8d}"
        )

    best_threshold, best_metrics = max(rows, key=lambda row: row[1]["dice"])

    print()
    print(
        "best validation Dice: "
        f"threshold={best_threshold:.2f} "
        f"dice={best_metrics['dice']:.4f} "
        f"iou={best_metrics['iou']:.4f} "
        f"precision={best_metrics['precision']:.4f} "
        f"recall={best_metrics['recall']:.4f}"
    )

    args.figure.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.plot(thresholds, [row[1]["dice"] for row in rows], label="Dice")
    axis.plot(thresholds, [row[1]["iou"] for row in rows], label="IoU")
    axis.plot(thresholds, [row[1]["precision"] for row in rows], label="Precision")
    axis.plot(thresholds, [row[1]["recall"] for row in rows], label="Recall")
    axis.set_xlabel("Segmentation threshold")
    axis.set_ylabel("Metric")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figure)
    plt.close(figure)

    print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
