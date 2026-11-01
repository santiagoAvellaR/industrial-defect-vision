import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class DeepLabV3ResNet50(nn.Module):
    """Binary semantic-segmentation wrapper around torchvision DeepLabV3-ResNet50."""

    def __init__(
        self,
        weights: DeepLabV3_ResNet50_Weights | None = DeepLabV3_ResNet50_Weights.DEFAULT,
    ) -> None:
        super().__init__()

        if weights is None:
            self.model = deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=1,
                aux_loss=False,
            )
        else:
            self.model = deeplabv3_resnet50(weights=weights)
            in_channels = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.model.aux_classifier = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["out"]
