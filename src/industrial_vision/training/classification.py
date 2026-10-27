import torch
from torch import nn


def _classification_counts(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, int, int]:
    predictions = predictions.long()
    targets = labels.long()

    tp = int(((predictions == 1) & (targets == 1)).sum().item())
    tn = int(((predictions == 0) & (targets == 0)).sum().item())
    fp = int(((predictions == 1) & (targets == 0)).sum().item())
    fn = int(((predictions == 0) & (targets == 1)).sum().item())
    return tp, tn, fp, fn


def _metrics_from_counts(
    total_samples: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": (tp + tn) / total_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def classification_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tp, tn, fp, fn = _classification_counts(predictions, labels)
    return _metrics_from_counts(labels.numel(), tp, tn, fp, fn)


def _epoch_metrics(
    total_loss: float,
    total_samples: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float | int]:
    metrics = _metrics_from_counts(total_samples, tp, tn, fp, fn)
    return {"loss": total_loss / total_samples, **metrics}


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    for images, _, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.detach() >= 0
        batch_tp, batch_tn, batch_fp, batch_fn = _classification_counts(predictions, labels)
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
) -> dict[str, float | int]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for images, _, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits >= 0
            batch_tp, batch_tn, batch_fp, batch_fn = _classification_counts(predictions, labels)
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn

    return _epoch_metrics(total_loss, total_samples, tp, tn, fp, fn)
