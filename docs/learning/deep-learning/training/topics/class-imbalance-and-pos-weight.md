# Class imbalance and `pos_weight`

EXP-001 showed that the unweighted binary classifier learned to predict every image as clean. This is a class-imbalance problem: the training split contains many more clean images than defective images.

For the training split:

```text
negative / clean = 1668
positive / defective = 197
```

A natural first positive-class weight is therefore:

```math
\mathrm{pos\_weight}
=
\frac{N_{\mathrm{negative}}}{N_{\mathrm{positive}}}
=
\frac{1668}{197}
\approx 8.47
```

## Unweighted BCE

For a binary target `y` and predicted probability `p`, binary cross-entropy is:

```math
L
=
-\left[y\log(p)+(1-y)\log(1-p)\right]
```

When positive examples are rare, the many negative examples can dominate the average loss. A model can therefore obtain a relatively low loss by predicting low defect probabilities for nearly every image.

## Weighted positive term

With `pos_weight = w`, the positive part of the loss is multiplied by `w`:

```math
L
=
-\left[w\,y\log(p)+(1-y)\log(1-p)\right]
```

For a negative sample (`y = 0`), the positive term disappears and the loss is unchanged.

For a positive sample (`y = 1`), its positive-class loss contribution is multiplied by `w`.

With `w ≈ 8.47`, missing a defective image produces a much stronger training signal than in EXP-001.

## Gradient intuition

For unweighted BCE with logits, the derivative with respect to logit `z` is:

```math
\frac{\partial L}{\partial z}=\sigma(z)-y
```

With positive weighting, for a positive example (`y = 1`):

```math
\frac{\partial L}{\partial z}
=
w\left(\sigma(z)-1\right)
```

For a negative example (`y = 0`):

```math
\frac{\partial L}{\partial z}
=
\sigma(z)
```

So `pos_weight` does not create more positive samples. It increases the loss and gradient contribution of positive samples already present in the batch.

## Why start with the negative-to-positive ratio?

Using:

```math
w=\frac{N_{\mathrm{negative}}}{N_{\mathrm{positive}}}
```

roughly balances the aggregate importance of the two classes in the loss when considering their frequencies.

It is a sensible baseline, not a universal optimum. Later experiments may tune the weight according to validation behavior and the practical cost of false negatives versus false positives.

## What stays unchanged in EXP-002

To isolate the effect of class weighting, keep the following fixed relative to EXP-001:

- same CNN;
- same train/validation split;
- same seed;
- same Adam optimizer;
- same learning rate;
- same batch size;
- same number of epochs;
- same classification threshold.

Change only the loss from unweighted `BCEWithLogitsLoss()` to `BCEWithLogitsLoss(pos_weight=...)`.

## Important caveat: weighting and probabilities

Class weighting changes the training objective. The resulting sigmoid outputs should not automatically be treated as perfectly calibrated real-world probabilities.

For EXP-002 we keep the same `0.5` classification threshold so that the comparison with EXP-001 is controlled. Threshold selection can be studied separately later rather than changing two variables in the same experiment.

## Mental model

`pos_weight` does **not** mean:

> predict more defects no matter what.

It means:

> errors on the rare positive class should influence learning more strongly.

The goal is to prevent the majority clean class from dominating the optimization signal.
