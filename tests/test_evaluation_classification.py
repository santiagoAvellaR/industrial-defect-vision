import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from industrial_vision.evaluation.classification import collect_classification_outputs


def test_collect_classification_outputs_returns_scores_labels_and_mask_areas() -> None:
    logits = torch.tensor([2.0, -2.0])
    masks = torch.tensor([
        [[[1.0, 0.0], [0.0, 0.0]]],
        [[[0.0, 0.0], [0.0, 0.0]]],
    ])
    labels = torch.tensor([1, 0])
    loader = DataLoader(TensorDataset(logits, masks, labels), batch_size=2)

    probabilities, output_labels, mask_area_ratios = collect_classification_outputs(
        nn.Identity(), loader, torch.device("cpu")
    )

    assert torch.allclose(probabilities, torch.sigmoid(logits))
    assert torch.equal(output_labels, labels)
    assert torch.allclose(mask_area_ratios, torch.tensor([0.25, 0.0]))
