# Concept — DeepLabV3-ResNet50 for binary defect segmentation

## What is it?

DeepLabV3 is a semantic-segmentation architecture. In this project it is used with a ResNet50 backbone and pretrained torchvision weights, then adapted to produce one binary defect logit per pixel.

The implemented model is therefore best described as:

`pretrained DeepLabV3 with a ResNet50 backbone for binary semantic segmentation`.

## Why do we need it in this project?

`SimpleUNet` gave us an interpretable scratch baseline for M4. M5 asks whether a stronger modern architecture with transfer learning can improve defect localization on the same KSDD2 split and loss.

The comparison is useful because it separates two modeling strategies:

- learn all visual features from scratch with a small U-Net;
- reuse a pretrained ResNet50 representation and adapt DeepLabV3 to industrial defects.

## Mental model

```text
RGB image
   ↓
ImageNet normalization
   ↓
ResNet50 backbone
   ↓
feature maps
   ↓
DeepLabV3 / ASPP
   ↓
binary segmentation head
   ↓
1 logit per pixel
   ↓
sigmoid
   ↓
probability map
   ↓
threshold
   ↓
binary defect mask
```

ResNet50 extracts increasingly abstract visual features. DeepLabV3 then uses atrous/dilated convolutions and ASPP to inspect those features at several effective spatial scales before producing a dense prediction.

## Technical details

### ResNet50 backbone

ResNet50 is a deep residual CNN. Residual blocks learn transformations while preserving shortcut paths, which makes deeper networks easier to optimize.

The backbone converts the input image into feature maps that encode edges, textures, shapes and higher-level patterns.

### Atrous or dilated convolutions

A dilated convolution inserts spacing between sampled kernel positions. This increases the receptive field without requiring equivalent extra downsampling.

That matters in segmentation because the model needs both local detail and enough surrounding context to decide whether a small irregular region is a defect.

### ASPP

ASPP means **Atrous Spatial Pyramid Pooling**. It processes the same feature map through parallel branches with different receptive fields, then combines them.

Mental model:

```text
features
  ├─ local convolution
  ├─ dilated convolution, smaller dilation
  ├─ dilated convolution, medium dilation
  ├─ dilated convolution, larger dilation
  └─ global-context branch
          ↓
       combine
```

This gives DeepLabV3 explicit multi-scale context.

### Binary head

The original pretrained model predicts multiple semantic classes. Our wrapper replaces the final classifier output with one channel so the contract matches the existing segmentation pipeline:

`[B, 3, H, W] -> [B, 1, H, W]`.

The output values are logits. `sigmoid` converts them to probabilities and a threshold converts probabilities to the final binary mask.

### Pretraining and normalization

The model starts from pretrained weights instead of random initialization. Because those weights expect ImageNet-style normalization, KSDD2 images are normalized before entering the network.

This preprocessing difference is part of using the pretrained representation correctly.

## SimpleUNet vs DeepLabV3-ResNet50

| Aspect | SimpleUNet | DeepLabV3-ResNet50 |
|---|---|---|
| Initialization | random | pretrained |
| Feature extractor | small custom encoder | ResNet50 |
| Decoder style | symmetric encoder-decoder with U-Net skips | DeepLabV3 dense head, not a symmetric U-Net decoder |
| Multi-scale context | limited | ASPP |
| Dilated convolutions | no | yes |
| BatchNorm | no | yes |
| Input normalization | `[0, 1]` tensor | ImageNet normalization |
| Compute cost | low | much higher |
| Learning strategy | from scratch | transfer learning / fine-tuning |

Both models still solve the same final problem: produce one defect logit per pixel.

## Where is it used in the repository?

- Model wrapper: `src/industrial_vision/models/deeplabv3.py`
- Training entry point: `scripts/train_deeplabv3.py`
- Shared loss and train/eval loop: `src/industrial_vision/training/segmentation.py`
- Checkpoint reconstruction: `src/industrial_vision/models/loading.py`
- Experiment: `docs/experiments/EXP-014-deeplabv3-resnet50-pretrained.md`

## Alternatives

- U-Net with a pretrained encoder;
- DeepLabV3 with another backbone;
- FCN;
- SegFormer or another transformer-based segmentation model;
- a larger custom U-Net.

These are not automatically better. Each changes capacity, compute cost, inductive bias and the amount of prior knowledge introduced.

## Trade-offs

Advantages:

- strong pretrained representation;
- explicit multi-scale context through ASPP;
- mature standard architecture;
- useful comparison against the scratch baseline.

Costs:

- much heavier than `SimpleUNet`;
- more VRAM and training time;
- BatchNorm introduces batch-size constraints;
- pretraining comes from natural images, so domain transfer is not guaranteed to help industrial defects.

## Common mistakes

- assuming a larger pretrained model must outperform a smaller baseline;
- forgetting the preprocessing expected by pretrained weights;
- comparing models with different validation splits;
- applying `sigmoid` before `BCEWithLogitsLoss`;
- treating the threshold as part of training instead of a separate decision rule;
- confusing U-Net skip connections with ResNet residual connections.

## Interview questions

1. Why can dilated convolutions help semantic segmentation?
2. What does ASPP add beyond a normal convolutional head?
3. Why is ImageNet normalization part of the DeepLabV3 transfer-learning setup?
4. How does DeepLabV3-ResNet50 differ conceptually from a U-Net?
5. Why does pretrained not necessarily mean better on industrial images?

## Evidence

- Code: `src/industrial_vision/models/deeplabv3.py`
- Experiment: `EXP-014`
- Smoke test: one full train/validation epoch completed successfully on the Colab T4 after fixing the singleton-batch issue.
