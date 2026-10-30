import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.loading import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save representative qualitative segmentation predictions."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("data/splits/val.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("artifacts/figures/segmentation_qualitative.png"),
    )
    return parser.parse_args()


def per_image_record(index: int, prediction: torch.Tensor, target: torch.Tensor) -> dict:
    prediction = prediction.bool()
    target = target.bool()

    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    gt_pixels = int(target.sum().item())
    pred_pixels = int(prediction.sum().item())
    denominator = 2 * tp + fp + fn
    dice = 2 * tp / denominator if denominator else 1.0

    return {
        "index": index,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
    }


def select_representative_cases(records: list[dict]) -> list[tuple[str, dict]]:
    selected = []

    detected = [row for row in records if row["gt_pixels"] > 0 and row["pred_pixels"] > 0]
    if detected:
        best = max(detected, key=lambda row: row["dice"])
        selected.append(("best overlap", best))

        remaining = [row for row in detected if row["index"] != best["index"]]
        if remaining:
            remaining = sorted(remaining, key=lambda row: row["dice"])
            selected.append(("partial defect", remaining[len(remaining) // 2]))

    false_negatives = [
        row for row in records if row["gt_pixels"] > 0 and row["pred_pixels"] == 0
    ]
    if false_negatives:
        selected.append(("false negative", max(false_negatives, key=lambda row: row["gt_pixels"])))

    false_positives = [
        row for row in records if row["gt_pixels"] == 0 and row["pred_pixels"] > 0
    ]
    if false_positives:
        selected.append(("false positive", max(false_positives, key=lambda row: row["pred_pixels"])))

    clean_correct = [
        row for row in records if row["gt_pixels"] == 0 and row["pred_pixels"] == 0
    ]
    if clean_correct:
        selected.append(("clean correct", clean_correct[0]))

    return selected


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_dataset = KSDD2Dataset(
        args.dataset_root,
        args.val_manifest,
        imagenet_normalize=checkpoint.get("imagenet_normalize", False),
    )
    display_dataset = KSDD2Dataset(args.dataset_root, args.val_manifest)
    loader = DataLoader(model_dataset, batch_size=args.batch_size, shuffle=False)

    records = []
    sample_index = 0

    with torch.inference_mode():
        for images, masks, _ in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu()
            predictions = probabilities >= args.threshold

            for prediction, target in zip(predictions, masks):
                records.append(per_image_record(sample_index, prediction, target))
                sample_index += 1

    selected = select_representative_cases(records)
    if not selected:
        raise RuntimeError("no representative validation cases were found")

    figure, axes = plt.subplots(len(selected), 4, figsize=(14, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    column_titles = ["Image", "Ground truth", "Probability map", "Prediction"]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title)

    with torch.inference_mode():
        for row_axes, (case_name, record) in zip(axes, selected):
            index = record["index"]
            model_image, mask, _ = model_dataset[index]
            display_image, _, _ = display_dataset[index]
            probability = torch.sigmoid(model(model_image.unsqueeze(0).to(device)))[0, 0].cpu()
            prediction = probability >= args.threshold
            sample = model_dataset.samples[index]

            row_axes[0].imshow(display_image.permute(1, 2, 0))
            row_axes[0].set_ylabel(
                f"{case_name}\nDice={record['dice']:.3f}\n{Path(sample['image']).name}",
                fontsize=8,
            )
            row_axes[1].imshow(mask[0], vmin=0, vmax=1)
            row_axes[2].imshow(probability, vmin=0, vmax=1)
            row_axes[3].imshow(prediction, vmin=0, vmax=1)

            for axis in row_axes:
                axis.axis("off")

    figure.suptitle(
        f"Segmentation qualitative cases — threshold={args.threshold:.2f}",
        fontsize=11,
    )
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=160)
    plt.close(figure)

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"threshold: {args.threshold:.2f}")
    for case_name, record in selected:
        sample = model_dataset.samples[record["index"]]
        print(
            f"{case_name}: {sample['image']} "
            f"dice={record['dice']:.4f} "
            f"gt_pixels={record['gt_pixels']} pred_pixels={record['pred_pixels']}"
        )
    print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
