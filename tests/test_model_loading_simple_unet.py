from industrial_vision.models.loading import build_model_from_checkpoint
from industrial_vision.models.simple_unet import SimpleUNet


def test_build_model_from_simple_unet_checkpoint() -> None:
    model = build_model_from_checkpoint({"model_name": "simple_unet"})

    assert isinstance(model, SimpleUNet)
