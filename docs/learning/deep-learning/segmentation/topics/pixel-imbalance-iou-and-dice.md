# Pixel imbalance, IoU and Dice

Semantic segmentation is evaluated at the pixel level, but ordinary pixel accuracy can be misleading when the positive class occupies only a small part of the image.

This is especially relevant for KSDD2 because many defects are small relative to the full image area.

## Pixel-level confusion matrix

The same TP/FP/FN/TN language used for classification can be applied to pixels.

For binary defect segmentation:

- **TP pixel**: ground truth says defect and prediction says defect;
- **FP pixel**: ground truth says background but prediction says defect;
- **FN pixel**: ground truth says defect but prediction says background;
- **TN pixel**: both say background.

The difference is the unit being counted.

Classification counts images. Segmentation counts pixels.

## Why pixel accuracy can be misleading

Suppose an image has 100,000 pixels and only 1,000 belong to the defect.

A useless model predicts background everywhere.

Then:

```text
TN = 99,000
FN = 1,000
TP = 0
FP = 0
```

Pixel accuracy is:

```math
\mathrm{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

which gives 99%.

But the model detected zero defect pixels.

So pixel accuracy is dominated by the abundant background class and is not enough for this project.

## Intersection and union

IoU and Dice focus on the predicted and true positive regions.

Let:

- `G` be the set of ground-truth defect pixels;
- `P` be the set of predicted defect pixels.

The **intersection** contains pixels marked as defect by both prediction and ground truth. In confusion-matrix language, its size is `TP`.

The **union** contains every pixel marked as defect by either prediction or ground truth. Its size is:

```text
TP + FP + FN
```

True-negative background pixels do not appear in the union.

## IoU: Intersection over Union

IoU is also called the Jaccard index.

```math
\mathrm{IoU}
=
\frac{TP}{TP + FP + FN}
```

Interpretation:

> Of all pixels that either the ground truth or prediction considered part of the defect, what fraction belongs to both?

Properties:

- perfect overlap: IoU = 1;
- no overlap: IoU = 0;
- false-positive and false-negative pixels both reduce IoU;
- true negatives do not inflate the score.

## Dice coefficient

Dice also measures overlap, but gives the intersection twice the weight:

```math
\mathrm{Dice}
=
\frac{2TP}{2TP + FP + FN}
```

Equivalent set form:

```math
\mathrm{Dice}
=
\frac{2|P \cap G|}{|P| + |G|}
```

Properties:

- perfect overlap: Dice = 1;
- no overlap: Dice = 0;
- Dice is usually numerically higher than IoU for the same non-perfect prediction;
- like IoU, it focuses on overlap rather than abundant true-negative background.

## Small numerical example

Suppose the true defect contains 6 pixels.

The model predicts 7 pixels as defect.

Of those, 5 overlap the real defect.

Then:

```text
TP = 5
FP = 2
FN = 1
```

IoU:

```math
\mathrm{IoU}
=
\frac{5}{5+2+1}
=
\frac{5}{8}
=0.625
```

Dice:

```math
\mathrm{Dice}
=
\frac{2\cdot5}{2\cdot5+2+1}
=
\frac{10}{13}
\approx0.769
```

Both describe the same overlap from slightly different normalizations.

## Relationship between Dice and IoU

For the same binary masks, they are mathematically related:

```math
\mathrm{Dice}
=
\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}
```

and therefore rank predictions consistently when computed from the same masks.

The project can report both because both are common in segmentation literature and interviews, but they should not be treated as independent evidence.

## Thresholding matters

The network outputs pixel logits:

```text
[B,1,H,W]
```

After sigmoid we obtain pixel defect scores.

A threshold converts those scores into a binary predicted mask.

Therefore IoU and Dice computed from a binary mask depend on the chosen threshold, just like classification precision/recall depended on the image-level threshold in M2.

We should distinguish:

- model scores/logits;
- threshold decision policy;
- overlap metrics after thresholding.

## Per-image vs global aggregation

There are multiple valid ways to aggregate segmentation metrics.

For learning, distinguish at least:

1. **global/micro overlap**: sum TP/FP/FN over all evaluated pixels, then compute IoU/Dice;
2. **per-image overlap**: compute a score for each image and then average.

These are not always identical, especially when defect sizes vary greatly.

The first M4 implementation should make the aggregation explicit rather than hiding it behind a library default.

## Empty-mask edge case

Clean images contain no positive ground-truth pixels.

If both ground truth and prediction are empty, the denominator of overlap formulas can become zero.

This is not merely a coding nuisance: we must define how clean images contribute to reported segmentation metrics.

The project should choose and document an explicit convention when the evaluation implementation is written.

Until then, do not silently invent a value.

## Mental model

```text
pixel accuracy
→ asks whether all pixels are classified correctly
→ can be dominated by background

IoU / Dice
→ ask how well predicted defect region overlaps real defect region
→ ignore abundant true-negative background in the core overlap formula
```

## Common mistakes

- Reporting only pixel accuracy on a highly imbalanced segmentation dataset.
- Forgetting that TP/FP/FN now refer to pixels, not images.
- Treating Dice and IoU as completely independent metrics.
- Computing binary IoU/Dice without stating the threshold.
- Ignoring the all-background/empty-mask case.
- Comparing metric values produced with different aggregation conventions without noticing the difference.

## Interview explanation

A concise explanation is:

> Pixel accuracy can be misleading in defect segmentation because most pixels are background. IoU and Dice instead measure overlap between the predicted defect region and the ground-truth defect region. IoU is TP divided by TP+FP+FN, while Dice is 2TP divided by 2TP+FP+FN. Both avoid letting the huge number of true-negative background pixels dominate the metric.

## Questions to be able to answer

1. What does TP mean in segmentation?
2. Why can a model with 99% pixel accuracy still be useless?
3. What is the intersection of predicted and ground-truth masks?
4. What is their union?
5. What is the IoU formula?
6. What is the Dice formula?
7. Why are true negatives absent from IoU and Dice formulas?
8. Why does threshold choice affect binary IoU and Dice?
9. What is the relationship between Dice and IoU?
10. Why must the empty-mask case be defined explicitly?
11. What is the difference between global and per-image aggregation?
