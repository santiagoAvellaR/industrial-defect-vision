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

Weighted BCE remains a controlled follow-up if simpler changes do not improve foreground recall.

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

### Empty masks

If an image has no foreground, then `y_i = 0` for every pixel. The per-image soft Dice becomes:

```math
\mathrm{SoftDice}_{empty}
=
\frac{s}{\sum_i p_i+s}
```

There is no foreground overlap to reward. The Dice term instead becomes small only when the predicted foreground probability mass approaches zero.

That is not mathematically invalid, but its effect matters when a dataset contains many clean images. If Dice is calculated independently for every image and then averaged, many empty masks can repeatedly push the model toward all-background predictions.

BCE already provides pixel-wise supervision on those clean images, so one controlled alternative is:

```text
clean image:
BCE  -> used
Dice -> skipped

image with defect:
BCE  -> used
Dice -> used
```

This is the hypothesis tested in EXP-011.

## 4. BCE + Dice

The first SimpleUNet experiment, EXP-010, used:

```math
\mathrm{Loss}
=
\mathrm{BCEWithLogitsLoss}
+
\mathrm{DiceLoss}
```

with Dice calculated for every target mask, including empty masks.

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

EXP-010 learned non-trivial defect localization but showed low foreground recall and strong epoch-to-epoch instability. A validation threshold sweep from `0.10` to `0.90` changed Dice only slightly, so threshold calibration was not the primary explanation.

EXP-011 therefore keeps BCE on every image but calculates the Dice term only for non-empty target masks. This isolates the treatment of empty masks without simultaneously changing the optimizer, learning rate, architecture or data split.

## Training versus evaluation

The training Dice term is **soft** and operates on sigmoid probabilities so gradients can flow.

Validation IoU and Dice are **hard metrics**:

```text
logits
-> sigmoid
-> threshold
-> binary mask
-> TP / FP / FN
-> IoU and Dice
```

Do not confuse Dice loss with the reported validation Dice metric. They are closely related conceptually but serve different purposes.

## Threshold sweep

A **threshold sweep** means evaluating the same trained model several times with different decision thresholds, without retraining it.

For example:

```text
same predicted probabilities
-> threshold 0.10 -> metrics
-> threshold 0.20 -> metrics
-> threshold 0.30 -> metrics
...
-> threshold 0.90 -> metrics
```

The model parameters do not change. Only the rule used to convert probabilities into a binary mask changes.

This lets us study the precision/recall trade-off and select a threshold on the validation split. A lower threshold usually predicts more foreground and tends to increase recall at the cost of more false positives. A higher threshold is more conservative and tends to increase precision while reducing recall.

The threshold must be selected using validation data, not the test set. Once selected, it should be frozen before final test evaluation.

In EXP-010 the sweep showed that changing the threshold barely affected Dice, so threshold choice was not the main problem. In EXP-011 and EXP-012, threshold tuning produced a modest additional improvement after the training changes had already improved the model itself.

## Training stability

Training is considered unstable when model behavior changes strongly from one epoch to the next instead of evolving in a reasonably consistent direction.

Normal training can contain some noise, so one temporary decrease is not enough to call training unstable. Strong evidence includes repeated large swings in validation metrics or prediction behavior.

EXP-011 showed this pattern. Consecutive epochs could move between very different operating points, for example from high precision and low recall to low precision and high recall, with large changes in false-positive pixels. Dice also improved, worsened and recovered repeatedly rather than following a smoother trend.

This motivated EXP-012, where the learning rate was reduced while keeping the architecture and loss formulation fixed.

## Checkpoint selection

EXP-010 selected only the lowest validation-loss checkpoint. The run showed that lowest validation loss and highest validation Dice can occur at different epochs.

From EXP-011 onward, SimpleUNet training preserves both:

```text
best validation loss checkpoint
best validation Dice checkpoint
```

This keeps optimization evidence and task-metric evidence separate instead of assuming they identify the same model state.

## Interview explanation

A concise explanation is:

> Pixel-level defect segmentation is extremely imbalanced because most pixels are background. BCE provides local supervision but can be dominated by background, while Dice focuses on foreground overlap. In the first U-Net run I also found that per-image Dice on many empty masks could add pressure toward all-background predictions. I therefore tested a controlled variant where BCE still supervises every image but Dice is applied only when the target mask contains foreground. I then used validation threshold sweeps to separate model-learning problems from decision-threshold effects, and reduced the learning rate when epoch-to-epoch metrics showed unstable optimization.

## Questions to be able to answer

1. Why can plain BCE be dominated by background pixels?
2. What does `pos_weight` change in weighted BCE?
3. Why does Dice not depend on true negatives?
4. Why must training Dice use probabilities rather than thresholded predictions?
5. What does per-image Dice become when the target mask is empty?
6. Why might skipping Dice on empty masks help while still keeping BCE there?
7. What complementary information do BCE and Dice contribute?
8. Why are Dice loss and validation Dice related but not identical?
9. Why can best validation loss and best validation Dice select different checkpoints?
10. What is a threshold sweep and why does it not require retraining the model?
11. Why should a threshold be selected on validation rather than test data?
12. What evidence distinguishes unstable training from normal epoch-to-epoch noise?
