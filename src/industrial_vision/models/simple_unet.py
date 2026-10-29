import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class SimpleUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder1 = _conv_block(3, 16)
        self.encoder2 = _conv_block(16, 32)
        self.encoder3 = _conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(64, 128)

        self.decoder3 = _conv_block(128 + 64, 64)
        self.decoder2 = _conv_block(64 + 32, 32)
        self.decoder1 = _conv_block(32 + 16, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(images)

        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))

        features = self.bottleneck(self.pool(skip3))

        features = F.interpolate(
            features,
            size=skip3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat((features, skip3), dim=1)
        features = self.decoder3(features)

        features = F.interpolate(
            features,
            size=skip2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat((features, skip2), dim=1)
        features = self.decoder2(features)

        features = F.interpolate(
            features,
            size=skip1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat((features, skip1), dim=1)
        features = self.decoder1(features)

        return self.output(features)
