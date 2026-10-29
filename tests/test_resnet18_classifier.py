import torch

from industrial_vision.models.resnet18_classifier import ResNet18Classifier


def test_resnet18_classifier_returns_one_logit_per_image() -> None:
    model = ResNet18Classifier(weights=None)
    images = torch.randn(2, 3, 224, 224)

    logits = model(images)

    assert logits.shape == (2,)


def test_frozen_resnet18_backbone_stays_frozen_in_train_mode() -> None:
    model = ResNet18Classifier(weights=None, trainable_backbone="none")
    model.train()

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert model.backbone.training is False
    assert model.classifier.training is True


def test_resnet18_can_fine_tune_only_layer4() -> None:
    model = ResNet18Classifier(weights=None, trainable_backbone="layer4")
    model.train()

    assert all(not parameter.requires_grad for parameter in model.backbone.layer1.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.layer4.parameters())
    assert model.backbone.layer1.training is False
    assert model.backbone.layer4.training is True
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
