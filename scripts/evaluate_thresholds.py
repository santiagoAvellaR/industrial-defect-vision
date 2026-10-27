import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.baseline_cnn import BaselineCNN
from industrial_vision.training.classification import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate classification thresholds on the validation split."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/splits/val.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("artifacts/figures/threshold_sweep.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = BaselineCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = KSDD2Dataset(args.dataset_root, args.val_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probabilities = []
    labels = []

    with torch.inference_mode():
        for images, _, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.sigmoid(logits).cpu())
            labels.append(batch_labels.cpu())

    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)

    thresholds = torch.linspace(0.10, 0.90, 17).tolist()
    rows = []

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"checkpoint val_loss: {checkpoint['val_loss']:.4f}")
    print()
    print("threshold precision recall   f1       tp   fp   fn")
    print("--------- --------- -------- -------- ---- ---- ----")

    for threshold in thresholds:
        metrics = classification_metrics(probabilities, labels, threshold)
        rows.append((threshold, metrics))
        print(
            f"{threshold:9.2f} "
            f"{metrics['precision']:9.4f} "
            f"{metrics['recall']:8.4f} "
            f"{metrics['f1']:8.4f} "
            f"{metrics['tp']:4d} "
            f"{metrics['fp']:4d} "
            f"{metrics['fn']:4d}"
        )

    best_threshold, best_metrics = max(rows, key=lambda row: row[1]["f1"])

    print()
    print(
        "best F1: "
        f"threshold={best_threshold:.2f} "
        f"precision={best_metrics['precision']:.4f} "
        f"recall={best_metrics['recall']:.4f} "
        f"f1={best_metrics['f1']:.4f}"
    )

    args.figure.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.plot(thresholds, [row[1]["precision"] for row in rows], label="Precision")
    axis.plot(thresholds, [row[1]["recall"] for row in rows], label="Recall")
    axis.plot(thresholds, [row[1]["f1"] for row in rows], label="F1")
    axis.set_xlabel("Classification threshold")
    axis.set_ylabel("Metric")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figure)
    plt.close(figure)

    print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
