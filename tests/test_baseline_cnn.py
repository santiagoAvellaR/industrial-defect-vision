import torch

from industrial_vision.models.baseline_cnn import BaselineCNN


def test_baseline_cnn_returns_one_logit_per_image() -> None:
    model = BaselineCNN()
    images = torch.randn(4, 3, 640, 224)

    logits = model(images)

    assert logits.shape == (4,)


def test_baseline_cnn_supports_global_max_pooling() -> None:
    model = BaselineCNN(global_pool="max")
    images = torch.randn(4, 3, 640, 224)

    logits = model(images)

    assert logits.shape == (4,)
