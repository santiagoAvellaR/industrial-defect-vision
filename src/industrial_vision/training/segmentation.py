import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss


def _segmentation_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    predictions = predictions.bool()
    targets = targets.bool()

    tp = int((predictions & targets).sum().item())
    tn = int((~predictions & ~targets).sum().item())
    fp = int((predictions & ~targets).sum().item())
    fn = int((~predictions & targets).sum().item())
    return tp, tn, fp, fn


def _metrics_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def segmentation_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tp, tn, fp, fn = _segmentation_counts(predictions, targets)
    return _metrics_from_counts(tp, tn, fp, fn)


def _epoch_metrics(
    total_loss: float,
    total_samples: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float | int]:
    metrics = _metrics_from_counts(tp, tn, fp, fn)
    return {"loss": total_loss / total_samples, **metrics}


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probabilities = torch.sigmoid(logits.detach())
        predictions = probabilities >= threshold
        batch_tp, batch_tn, batch_fp, batch_fn = _segmentation_counts(predictions, masks)
        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn

    return _epoch_metrics(total_loss, total_samples, tp, tn, fp, fn)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= threshold
            batch_tp, batch_tn, batch_fp, batch_fn = _segmentation_counts(predictions, masks)
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn

    return _epoch_metrics(total_loss, total_samples, tp, tn, fp, fn)
