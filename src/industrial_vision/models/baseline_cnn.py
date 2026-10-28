import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, global_pool: str = "avg") -> None:
        super().__init__()
        if global_pool not in {"avg", "max", "avgmax"}:
            raise ValueError("global_pool must be 'avg', 'max' or 'avgmax'")

        self.global_pool = global_pool
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

        classifier_features = 128 if global_pool == "avgmax" else 64
        self.classifier = nn.Linear(classifier_features, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)

        if self.global_pool == "avg":
            pooled = features.mean(dim=(2, 3))
        elif self.global_pool == "max":
            pooled = features.amax(dim=(2, 3))
        else:
            average = features.mean(dim=(2, 3))
            maximum = features.amax(dim=(2, 3))
            pooled = torch.cat((average, maximum), dim=1)

        return self.classifier(pooled).squeeze(1)
