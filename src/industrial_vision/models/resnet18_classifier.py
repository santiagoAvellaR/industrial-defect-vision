import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(
        self,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = resnet18(weights=weights)

        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feature_count, 1)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features).squeeze(1)
