import torch

from industrial_vision.models.resnet18_classifier import ResNet18Classifier


def test_resnet18_classifier_returns_one_logit_per_image() -> None:
    model = ResNet18Classifier(weights=None)
    images = torch.randn(2, 3, 224, 224)

    logits = model(images)

    assert logits.shape == (2,)


def test_frozen_resnet18_backbone_stays_frozen_in_train_mode() -> None:
    model = ResNet18Classifier(weights=None, freeze_backbone=True)
    model.train()

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert model.backbone.training is False
    assert model.classifier.training is True
