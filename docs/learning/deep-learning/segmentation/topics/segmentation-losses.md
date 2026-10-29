# Segmentation losses: BCE, weighted BCE and Dice

A segmentation model produces one logit at every spatial position:

```text
logits  [B, 1, H, W]
targets [B, 1, H, W]
```

The loss must turn the difference between those predicted pixel logits and the ground-truth mask into one scalar used by backpropagation.

KSDD2 has a severe pixel-level imbalance: most pixels are background and only a small fraction belong to defects. This makes the loss choice important.

## 1. Binary cross entropy

`BCEWithLogitsLoss` applies binary cross entropy to every pixel while receiving raw logits.

For one target pixel `y` and predicted probability `p`, binary cross entropy is:

```math
\mathrm{BCE}(p,y)
=
-\left[y\log(p) + (1-y)\log(1-p)\right]
```

PyTorch's `BCEWithLogitsLoss` combines the sigmoid and BCE calculation in a numerically stable implementation, so the model should return logits directly.

### Strength

BCE gives a clear local learning signal at every pixel, including clean/background regions.

### Problem with severe imbalance

If almost every target pixel is background, the total BCE can be dominated by background pixels. A model can reduce loss substantially by becoming very confident about background while still doing poorly on the small foreground defect.

This is analogous to the image-level imbalance observed during M2, but now the imbalance is between pixels instead of images.

## 2. Weighted BCE

Weighted BCE increases the contribution of positive/defect pixels.

In PyTorch this can be done with the `pos_weight` argument of `BCEWithLogitsLoss`.

Conceptually:

```text
background pixel error -> normal contribution
defect pixel error     -> multiplied contribution
```

A common data-driven starting point is related to the ratio of negative pixels to positive pixels.

### Strength

It directly counteracts the foreground/background imbalance while preserving the simple pixel-wise BCE objective.

### Trade-off

For datasets with extremely small defects, the pixel ratio can make `pos_weight` very large. That value becomes an important hyperparameter and can push the model toward excessive foreground prediction if chosen poorly.

For the first M4 baseline we therefore do not introduce this extra variable yet. It remains a controlled follow-up experiment if the first model under-detects foreground.

## 3. Dice coefficient and Dice loss

The hard Dice metric for binary masks is:

```math
\mathrm{Dice}
=
\frac{2TP}{2TP+FP+FN}
```

It focuses on foreground overlap and does not contain true negatives. That makes it useful when the background is overwhelmingly large.

To train with gradient descent, a differentiable or **soft Dice** version uses sigmoid probabilities instead of thresholded binary predictions.

For one image, conceptually:

```math
\mathrm{SoftDice}
=
\frac{2\sum_i p_i y_i + s}
{\sum_i p_i + \sum_i y_i + s}
```

where:

- `p_i` is the predicted defect probability at pixel `i`;
- `y_i` is the binary target mask at pixel `i`;
- `s` is a small smoothing constant that avoids unstable zero denominators.

Dice loss is:

```math
\mathrm{DiceLoss}=1-\mathrm{SoftDice}
```

### Strength

Dice directly rewards overlap with the small foreground region and is not dominated by the enormous number of correctly predicted background pixels.

### Limitation of pure Dice loss

Pure Dice gives a more global overlap signal rather than an independent local penalty at each pixel. Empty masks also require careful handling because there is no foreground intersection to measure.

For a dataset containing many clean images, using Dice alone is therefore not the simplest first choice.

## 4. BCE + Dice: first M4 baseline

The first SimpleUNet experiment uses:

```math
\mathrm{Loss}
=
\mathrm{BCEWithLogitsLoss}
+
\mathrm{DiceLoss}
```

The implementation is in:

`src/industrial_vision/training/segmentation.py`

The two terms have complementary roles:

```text
BCE
-> local pixel-by-pixel supervision
-> explicitly penalizes false foreground on clean/background pixels

Dice
-> foreground overlap supervision
-> reduces the dominance of true-negative background pixels
```

This is not assumed to be universally optimal. It is a deliberately simple starting point that addresses the known imbalance without adding a pixel `pos_weight` hyperparameter yet.

## Training versus evaluation

The training Dice term is **soft** and operates on sigmoid probabilities so gradients can flow.

Validation IoU and Dice are **hard metrics**:

```text
logits
-> sigmoid
-> threshold (default 0.5)
-> binary mask
-> TP / FP / FN
-> IoU and Dice
```

Do not confuse Dice loss with the reported validation Dice metric. They are closely related conceptually but serve different purposes.

## Current baseline decision

For the first M4 segmentation run:

```text
model      = SimpleUNet
loss       = BCE + Dice
threshold  = 0.5
checkpoint = lowest validation loss
```

If error analysis later shows poor foreground recall, weighted BCE is a natural controlled experiment. If boundaries or tiny defects remain difficult, the next change should be justified from predictions rather than added automatically.

## Interview explanation

A concise explanation is:

> Pixel-level defect segmentation is extremely imbalanced because most pixels are background. Plain BCE provides useful local supervision but can be dominated by background. Dice focuses directly on foreground overlap and ignores true negatives. For the first U-Net baseline I combined BCE and soft Dice so training receives both local pixel-wise supervision and an overlap-based signal, while keeping weighted BCE as a later controlled experiment.

## Questions to be able to answer

1. Why can plain BCE be dominated by background pixels?
2. What does `pos_weight` change in weighted BCE?
3. Why does Dice not depend on true negatives?
4. Why must training Dice use probabilities rather than thresholded predictions?
5. Why can pure Dice be awkward for empty masks?
6. What complementary information do BCE and Dice contribute?
7. Why are Dice loss and validation Dice related but not identical?
