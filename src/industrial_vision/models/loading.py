from torch import nn

from industrial_vision.models.baseline_cnn import BaselineCNN
from industrial_vision.models.resnet18_classifier import ResNet18Classifier


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    model_name = checkpoint.get("model_name", "baseline_cnn")

    if model_name == "baseline_cnn":
        return BaselineCNN(global_pool=checkpoint.get("global_pool", "avg"))

    if model_name == "resnet18":
        return ResNet18Classifier(
            weights=None,
            freeze_backbone=checkpoint.get("freeze_backbone", False),
        )

    raise ValueError(f"unknown model_name: {model_name}")
