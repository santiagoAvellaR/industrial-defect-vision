import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from industrial_vision.training.classification import (
    classification_metrics,
    evaluate,
    train_one_epoch,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images).squeeze(1)


def test_train_one_epoch_updates_model_parameters() -> None:
    images = torch.tensor([[-1.0], [1.0], [-2.0], [2.0]])
    masks = torch.zeros(4, 1)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, masks, labels), batch_size=2)

    model = TinyClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.linear.weight.detach().clone()

    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert not torch.equal(before, model.linear.weight.detach())
    assert metrics["loss"] >= 0
    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == 4


def test_evaluate_reports_binary_metrics_without_updating_parameters() -> None:
    logits = torch.tensor([2.0, -2.0, -1.0, 1.0])
    masks = torch.zeros(4, 1)
    labels = torch.tensor([1, 0, 1, 0])
    loader = DataLoader(TensorDataset(logits, masks, labels), batch_size=2)

    model = nn.Identity()
    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1


def test_lower_threshold_increases_recall_for_fixed_probabilities() -> None:
    probabilities = torch.tensor([0.9, 0.6, 0.4, 0.2])
    labels = torch.tensor([1, 0, 1, 0])

    default_metrics = classification_metrics(probabilities, labels, threshold=0.5)
    lower_metrics = classification_metrics(probabilities, labels, threshold=0.3)

    assert default_metrics["recall"] == 0.5
    assert lower_metrics["recall"] == 1.0
    assert lower_metrics["fp"] > default_metrics["fp"]
