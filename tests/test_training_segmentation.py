import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from industrial_vision.training.segmentation import (
    BCEDiceLoss,
    evaluate,
    segmentation_metrics,
    train_one_epoch,
)


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images)


def test_segmentation_metrics_count_pixel_overlap() -> None:
    probabilities = torch.tensor([[[[0.9, 0.8], [0.2, 0.1]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    metrics = segmentation_metrics(probabilities, targets, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert abs(float(metrics["iou"]) - 1 / 3) < 1e-6
    assert metrics["dice"] == 0.5


def test_bce_dice_loss_prefers_better_logits() -> None:
    targets = torch.tensor([[[[1.0, 0.0]]]])
    good_logits = torch.tensor([[[[8.0, -8.0]]]])
    bad_logits = torch.tensor([[[[-8.0, 8.0]]]])
    criterion = BCEDiceLoss()

    assert criterion(good_logits, targets) < criterion(bad_logits, targets)


def test_positive_only_dice_uses_bce_for_empty_masks() -> None:
    logits = torch.tensor([[[[1.0, -1.0], [0.5, -0.5]]]])
    targets = torch.zeros_like(logits)
    criterion = BCEDiceLoss(dice_on_positive_masks_only=True)
    expected = nn.BCEWithLogitsLoss()(logits, targets)

    assert torch.allclose(criterion(logits, targets), expected)


def test_train_one_epoch_updates_segmentation_model() -> None:
    images = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ]
    )
    masks = images.clone()
    labels = torch.tensor([1, 1])
    loader = DataLoader(TensorDataset(images, masks, labels), batch_size=1)

    model = TinySegmenter()
    criterion = BCEDiceLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.conv.weight.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        non_blocking=True,
    )

    assert not torch.equal(before, model.conv.weight.detach())
    assert metrics["loss"] >= 0


def test_evaluate_does_not_update_segmentation_model() -> None:
    images = torch.zeros(2, 1, 2, 2)
    masks = torch.zeros(2, 1, 2, 2)
    labels = torch.tensor([0, 0])
    loader = DataLoader(TensorDataset(images, masks, labels), batch_size=1)

    model = TinySegmenter()
    criterion = BCEDiceLoss()
    before = model.conv.weight.detach().clone()

    metrics = evaluate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        non_blocking=True,
    )

    assert torch.equal(before, model.conv.weight.detach())
    assert metrics["loss"] >= 0
