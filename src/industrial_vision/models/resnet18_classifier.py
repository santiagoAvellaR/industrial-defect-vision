import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(
        self,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        freeze_backbone: bool = True,
        trainable_backbone: str | None = None,
    ) -> None:
        super().__init__()

        if trainable_backbone is None:
            trainable_backbone = "none" if freeze_backbone else "all"

        if trainable_backbone not in {"none", "layer4", "all"}:
            raise ValueError(
                "trainable_backbone must be 'none', 'layer4', or 'all'"
            )

        self.trainable_backbone = trainable_backbone
        self.freeze_backbone = trainable_backbone == "none"
        self.backbone = resnet18(weights=weights)

        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feature_count, 1)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if trainable_backbone == "layer4":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        elif trainable_backbone == "all":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)

        if mode and self.trainable_backbone == "none":
            self.backbone.eval()
        elif mode and self.trainable_backbone == "layer4":
            self.backbone.eval()
            self.backbone.layer4.train()

        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features).squeeze(1)
