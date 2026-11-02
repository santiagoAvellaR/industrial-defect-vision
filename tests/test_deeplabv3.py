import torch

from industrial_vision.models.deeplabv3 import DeepLabV3ResNet50
from industrial_vision.models.loading import build_model_from_checkpoint


def test_deeplabv3_binary_output_shape() -> None:
    model = build_model_from_checkpoint({"model_name": "deeplabv3_resnet50"})
    assert isinstance(model, DeepLabV3ResNet50)
    model.eval()

    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (1, 1, 64, 64)
