# Semantic segmentation

This section records the concepts used in M4, where the project moves from image-level defect classification to pixel-level defect localization.

## Reading order

1. [Encoder, decoder, bottleneck and U-Net skip connections](topics/encoder-decoder-and-unet.md)
2. [Pixel logits](topics/pixel-logits.md) — how the network produces one learned defect logit at every spatial position.
3. [Upsampling](topics/upsampling.md) — how a decoder increases spatial resolution and why enlargement alone does not restore lost detail.
4. [Pixel imbalance, IoU and Dice](topics/pixel-imbalance-iou-and-dice.md) — why pixel accuracy is misleading and how overlap metrics evaluate defect masks.
5. [Simple U-Net forward pass](topics/simple-unet-forward-pass.md) — the first implemented M4 model, followed tensor by tensor through encoder, bottleneck, decoder and skips.

For a cross-project visual summary of every architecture studied so far, see the [architecture atlas](../architectures/README.md).

## Core mental model

Classification answers whether an image contains a defect. Semantic segmentation assigns a class to each pixel.

For binary defect segmentation, the model should transform an image tensor shaped `[B, 3, H, W]` into one defect logit per pixel shaped `[B, 1, H, W]`.

```mermaid
flowchart LR
    A[Image B×3×H×W] --> B[Segmentation model]
    B --> C[Pixel logits B×1×H×W]
    C --> D[Sigmoid]
    D --> E[Defect score per pixel]
    E --> F[Threshold]
    F --> G[Binary defect mask]
```

The first implemented M4 model is `SimpleUNet` in:

`src/industrial_vision/models/simple_unet.py`

It deliberately uses a small scratch encoder-decoder so the complete segmentation forward pass can be understood before moving to a modern pretrained segmentation model in M5.

## M4 questions

By the end of M4, we should be able to answer:

- Why does segmentation need spatial output instead of a single logit?
- What is the role of the encoder?
- What is the role of the decoder?
- What is the bottleneck?
- Why can downsampling hurt localization?
- What does upsampling do, and what can it not recover by itself?
- Why does U-Net copy high-resolution encoder features into the decoder?
- How are U-Net skip connections different from ResNet residual connections?
- How does the final layer produce one logit per pixel?
- What shape should a binary segmentation model output?
- Why can pixel accuracy be misleading for small defects?
- What do IoU and Dice measure?
- How do pixel-level losses and metrics differ from image-level classification metrics?
- Can we follow the implemented U-Net shapes through every encoder and decoder stage?
