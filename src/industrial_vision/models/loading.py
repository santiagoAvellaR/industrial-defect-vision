from torch import nn

from industrial_vision.models.baseline_cnn import BaselineCNN
from industrial_vision.models.resnet18_classifier import ResNet18Classifier


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    model_name = checkpoint.get("model_name", "baseline_cnn")

    if model_name == "baseline_cnn":
        return BaselineCNN(global_pool=checkpoint.get("global_pool", "avg"))

    if model_name == "resnet18":
        trainable_backbone = checkpoint.get("trainable_backbone")
        if trainable_backbone is None:
            trainable_backbone = (
                "none" if checkpoint.get("freeze_backbone", False) else "all"
            )

        return ResNet18Classifier(
            weights=None,
            trainable_backbone=trainable_backbone,
        )

    raise ValueError(f"unknown model_name: {model_name}")
