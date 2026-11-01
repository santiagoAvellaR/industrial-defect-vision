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
    parser.add_argument(
        "--latest-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint overwritten after every completed epoch for interruption recovery.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a compatible checkpoint. --epochs remains the total target epoch count.",
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


def _validate_resume_checkpoint(
    checkpoint: dict,
    *,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    seed: int,
    dice_positive_only: bool,
) -> None:
    expected = {
        "model_name": "deeplabv3_resnet50",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "threshold": threshold,
        "seed": seed,
        "dice_on_positive_masks_only": dice_positive_only,
        "imagenet_normalize": True,
        "train_drop_last": True,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value!r}, requested={expected_value!r}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Resume checkpoint is not compatible with this run: {details}")


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.batch_size < 2:
        raise ValueError(
            "--batch-size must be >= 2 for DeepLabV3 BatchNorm during training"
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

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
        f"num_workers={args.num_workers} pin_memory={pin_memory} "
        "train_drop_last=True"
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

    dropped_train_samples = len(train_dataset) % args.batch_size
    print(f"train samples dropped per epoch: {dropped_train_samples}")

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
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model_weights = None if args.resume is not None else weights
    model = DeepLabV3ResNet50(weights=model_weights).to(device)
    criterion = BCEDiceLoss(dice_on_positive_masks_only=args.dice_positive_only)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_dice = float("-inf")
    start_epoch = 1

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.dice_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.latest_checkpoint is not None:
        args.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        _validate_resume_checkpoint(
            resume_checkpoint,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            seed=args.seed,
            dice_positive_only=args.dice_positive_only,
        )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val_loss = float(
            resume_checkpoint.get("best_val_loss", resume_checkpoint["val_loss"])
        )
        best_val_dice = float(
            resume_checkpoint.get(
                "best_val_dice",
                resume_checkpoint["val_metrics"]["dice"],
            )
        )

        if "data_loader_generator_state" in resume_checkpoint:
            generator.set_state(resume_checkpoint["data_loader_generator_state"])
        if "torch_rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in resume_checkpoint:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])

        print(f"resuming from: {args.resume}")
        print(f"resume checkpoint epoch: {resume_checkpoint['epoch']}")
        print(f"next epoch: {start_epoch}/{args.epochs}")

        if start_epoch > args.epochs:
            print("Checkpoint already reached the requested total epoch count. Nothing to do.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
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

        val_loss = float(val_metrics["loss"])
        val_dice = float(val_metrics["dice"])
        is_best_loss = val_loss < best_val_loss
        is_best_dice = val_dice > best_val_dice

        if is_best_loss:
            best_val_loss = val_loss
        if is_best_dice:
            best_val_dice = val_dice

        checkpoint = {
            "epoch": epoch,
            "model_name": "deeplabv3_resnet50",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
            "loss_name": loss_name,
            "dice_on_positive_masks_only": args.dice_positive_only,
            "threshold": args.threshold,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "train_drop_last": True,
            "dropped_train_samples_per_epoch": dropped_train_samples,
            "device_type": device.type,
            "device_name": device_name,
            "pretrained": True,
            "pretrained_weights": weights_name,
            "imagenet_normalize": True,
            "data_loader_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        }

        if is_best_loss:
            torch.save(checkpoint, args.checkpoint)
            print(f"saved best-loss checkpoint: {args.checkpoint}")

        if is_best_dice:
            torch.save(checkpoint, args.dice_checkpoint)
            print(f"saved best-Dice checkpoint: {args.dice_checkpoint}")

        if args.latest_checkpoint is not None:
            torch.save(checkpoint, args.latest_checkpoint)
            print(f"saved latest checkpoint: {args.latest_checkpoint}")


if __name__ == "__main__":
    main()
