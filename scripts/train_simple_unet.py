import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.simple_unet import SimpleUNet
from industrial_vision.training.segmentation import BCEDiceLoss, evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M4 simple U-Net segmentation baseline.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dice-positive-only",
        action="store_true",
        help="Compute the Dice term only for target masks that contain foreground.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_unet_bce_dice.pt"),
        help="Checkpoint selected by lowest validation loss.",
    )
    parser.add_argument(
        "--dice-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint selected by highest validation Dice.",
    )
    return parser.parse_args()


def print_metrics(split: str, metrics: dict[str, float | int]) -> None:
    print(
        f"{split}: "
        f"loss={metrics['loss']:.4f} "
        f"iou={metrics['iou']:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_name = "bce_dice_positive_masks_only" if args.dice_positive_only else "bce_dice"
    dice_checkpoint = args.dice_checkpoint
    if dice_checkpoint is None:
        stem = args.checkpoint.stem
        if stem.endswith("_best_loss"):
            stem = stem.removesuffix("_best_loss") + "_best_dice"
        else:
            stem = stem + "_best_dice"
        dice_checkpoint = args.checkpoint.with_name(stem + args.checkpoint.suffix)

    print(f"device: {device}")
    print("model: simple_unet")
    print(
        "loss: bce + dice (positive masks only)"
        if args.dice_positive_only
        else "loss: bce + dice"
    )
    print(f"threshold: {args.threshold}")

    train_dataset = KSDD2Dataset(args.dataset_root, args.train_manifest)
    val_dataset = KSDD2Dataset(args.dataset_root, args.val_manifest)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleUNet().to(device)
    criterion = BCEDiceLoss(dice_on_positive_masks_only=args.dice_positive_only)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_dice = float("-inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dice_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold=args.threshold,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=args.threshold,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_metrics("train", train_metrics)
        print_metrics("val  ", val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_name": "simple_unet",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_metrics["loss"]),
            "val_metrics": val_metrics,
            "loss_name": loss_name,
            "dice_on_positive_masks_only": args.dice_positive_only,
            "threshold": args.threshold,
            "seed": args.seed,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(checkpoint, args.checkpoint)
            print(f"saved best-loss checkpoint: {args.checkpoint}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = float(val_metrics["dice"])
            torch.save(checkpoint, dice_checkpoint)
            print(f"saved best-Dice checkpoint: {dice_checkpoint}")


if __name__ == "__main__":
    main()
