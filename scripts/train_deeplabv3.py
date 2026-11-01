import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.deeplabv3 import DeepLabV3ResNet50
from industrial_vision.training.segmentation import BCEDiceLoss, evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the M5 pretrained DeepLabV3-ResNet50 segmentation model."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--dice-positive-only",
        action="store_true",
        help="Compute the Dice term only for target masks that contain foreground.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/deeplabv3_resnet50_best_loss.pt"),
        help="Checkpoint selected by lowest validation loss.",
    )
    parser.add_argument(
        "--dice-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/deeplabv3_resnet50_best_dice.pt"),
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
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = args.pin_memory and device.type == "cuda"
    non_blocking = pin_memory
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    weights_name = weights.name
    loss_name = (
        "bce_dice_positive_masks_only" if args.dice_positive_only else "bce_dice"
    )

    print(f"device: {device}")
    print(f"device name: {device_name}")
    print("model: deeplabv3_resnet50")
    print(f"pretrained weights: {weights_name}")
    print("imagenet normalization: True")
    print(
        "loss: bce + dice (positive masks only)"
        if args.dice_positive_only
        else "loss: bce + dice"
    )
    print(f"threshold: {args.threshold}")
    print(
        f"data loader: batch_size={args.batch_size} "
        f"num_workers={args.num_workers} pin_memory={pin_memory}"
    )

    train_dataset = KSDD2Dataset(
        args.dataset_root,
        args.train_manifest,
        imagenet_normalize=True,
    )
    val_dataset = KSDD2Dataset(
        args.dataset_root,
        args.val_manifest,
        imagenet_normalize=True,
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = DeepLabV3ResNet50(weights=weights).to(device)
    criterion = BCEDiceLoss(dice_on_positive_masks_only=args.dice_positive_only)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_dice = float("-inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.dice_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold=args.threshold,
            non_blocking=non_blocking,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=args.threshold,
            non_blocking=non_blocking,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_metrics("train", train_metrics)
        print_metrics("val  ", val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_name": "deeplabv3_resnet50",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_metrics["loss"]),
            "val_metrics": val_metrics,
            "loss_name": loss_name,
            "dice_on_positive_masks_only": args.dice_positive_only,
            "threshold": args.threshold,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "device_type": device.type,
            "device_name": device_name,
            "pretrained": True,
            "pretrained_weights": weights_name,
            "imagenet_normalize": True,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(checkpoint, args.checkpoint)
            print(f"saved best-loss checkpoint: {args.checkpoint}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = float(val_metrics["dice"])
            torch.save(checkpoint, args.dice_checkpoint)
            print(f"saved best-Dice checkpoint: {args.dice_checkpoint}")


if __name__ == "__main__":
    main()
