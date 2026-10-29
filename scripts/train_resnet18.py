import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from industrial_vision.data.dataset import KSDD2Dataset
from industrial_vision.models.resnet18_classifier import ResNet18Classifier
from industrial_vision.training.classification import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a pretrained ResNet18 classifier."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--trainable-backbone",
        choices=("none", "layer4", "all"),
        default="none",
    )
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/resnet18_frozen.pt"),
    )
    return parser.parse_args()


def print_metrics(split: str, metrics: dict[str, float | int]) -> None:
    print(
        f"{split}: "
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"tp={metrics['tp']} tn={metrics['tn']} "
        f"fp={metrics['fp']} fn={metrics['fn']}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("model: resnet18 pretrained")
    print(f"trainable backbone: {args.trainable_backbone}")
    print("imagenet normalization: enabled")

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.initial_checkpoint is None:
        model = ResNet18Classifier(
            trainable_backbone=args.trainable_backbone
        ).to(device)
    else:
        model = ResNet18Classifier(
            weights=None,
            trainable_backbone=args.trainable_backbone,
        ).to(device)
        initial_checkpoint = torch.load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"initial checkpoint: {args.initial_checkpoint}")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"trainable parameters: {trainable_parameters:,}")

    if args.pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
        print("pos_weight: none")
    else:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"pos_weight: {args.pos_weight}")

    parameter_groups = [
        {
            "params": model.classifier.parameters(),
            "lr": args.learning_rate,
        }
    ]

    if args.trainable_backbone == "layer4":
        parameter_groups.append(
            {
                "params": model.backbone.layer4.parameters(),
                "lr": args.backbone_learning_rate,
            }
        )
    elif args.trainable_backbone == "all":
        parameter_groups.append(
            {
                "params": model.backbone.parameters(),
                "lr": args.backbone_learning_rate,
            }
        )

    optimizer = torch.optim.Adam(parameter_groups)
    print(f"classifier learning rate: {args.learning_rate}")
    if args.trainable_backbone != "none":
        print(f"backbone learning rate: {args.backbone_learning_rate}")

    best_val_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_metrics("train", train_metrics)
        print_metrics("val  ", val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "resnet18",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "pos_weight": args.pos_weight,
                    "trainable_backbone": args.trainable_backbone,
                    "freeze_backbone": args.trainable_backbone == "none",
                    "imagenet_normalize": True,
                    "classifier_learning_rate": args.learning_rate,
                    "backbone_learning_rate": args.backbone_learning_rate,
                    "initial_checkpoint": (
                        str(args.initial_checkpoint)
                        if args.initial_checkpoint is not None
                        else None
                    ),
                    "seed": args.seed,
                },
                args.checkpoint,
            )
            print(f"saved best checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
