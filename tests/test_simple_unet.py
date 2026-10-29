import torch

from industrial_vision.models.simple_unet import SimpleUNet


def test_simple_unet_returns_one_logit_per_pixel() -> None:
    model = SimpleUNet()
    images = torch.randn(2, 3, 640, 224)

    logits = model(images)

    assert logits.shape == (2, 1, 640, 224)
