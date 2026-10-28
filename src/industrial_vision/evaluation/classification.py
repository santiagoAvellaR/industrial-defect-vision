import torch
from torch import nn


def collect_classification_outputs(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    probabilities = []
    labels = []
    mask_area_ratios = []

    with torch.inference_mode():
        for images, masks, batch_labels in loader:
            logits = model(images.to(device))

            probabilities.append(torch.sigmoid(logits).cpu())
            labels.append(batch_labels.cpu())
            mask_area_ratios.append(masks.mean(dim=(1, 2, 3)).cpu())

    return (
        torch.cat(probabilities),
        torch.cat(labels),
        torch.cat(mask_area_ratios),
    )
