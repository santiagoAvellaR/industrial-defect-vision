# Classification metrics for imbalanced defect detection

These metrics describe the behavior of a binary classifier after converting its output into a predicted class.

For this project:

- positive class = defective image (`1`);
- negative class = clean image (`0`).

## Confusion-matrix terms

| Term | Meaning | Industrial interpretation |
|---|---|---|
| TP — true positive | predicted defective, actually defective | defect correctly detected |
| TN — true negative | predicted clean, actually clean | clean sample correctly accepted |
| FP — false positive | predicted defective, actually clean | false alarm / unnecessary rejection |
| FN — false negative | predicted clean, actually defective | missed defect |

A useful mental rule is:

> **True/False tells us whether the prediction was correct. Positive/Negative tells us what the model predicted.**

So:

- `FP`: the model predicted **positive / defective**, but that prediction was false;
- `FN`: the model predicted **negative / clean**, but that prediction was false.

In industrial inspection, false negatives are often particularly important because a defective product may pass inspection.

## Accuracy

```math
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

Accuracy measures the fraction of all predictions that are correct.

It can be misleading when classes are imbalanced. In KSDD2, most images are clean, so predicting every image as clean already gives about 89.5% validation accuracy.

## Precision

```math
\mathrm{Precision} = \frac{TP}{TP + FP}
```

Precision answers:

> Of all images predicted as defective, how many were actually defective?

High precision means few false alarms.

## Recall

```math
\mathrm{Recall} = \frac{TP}{TP + FN}
```

Recall answers:

> Of all truly defective images, how many did the model detect?

High recall means few missed defects.

For industrial defect detection, recall is often especially important because an FN corresponds to a real defect that the classifier failed to detect.

## F1 score

```math
F_1 = 2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
```

F1 combines precision and recall through their harmonic mean.

It becomes low if either precision or recall is low, so it is useful when a single number is needed to summarize positive-class detection under class imbalance.

## Example from EXP-001

Validation results for the unweighted CNN baseline were:

```text
TP = 0
TN = 417
FP = 0
FN = 49
```

Therefore:

- accuracy = `417 / 466 ≈ 0.895`;
- recall = `0 / 49 = 0`;
- F1 = `0`;
- the model predicted every sample as clean.

This is why accuracy alone would have been misleading.

## Mental checklist

When reading a classification experiment, ask:

1. How many real defects were detected? → TP
2. How many real defects were missed? → FN
3. How many clean images triggered false alarms? → FP
4. How many clean images were correctly accepted? → TN
5. Among predicted defects, how trustworthy are they? → precision
6. Among real defects, how many did we catch? → recall
7. Is there a balanced single-number summary of positive-class performance? → F1
