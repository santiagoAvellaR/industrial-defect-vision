import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = features.mean(dim=(2, 3))
        return self.classifier(pooled).squeeze(1)
