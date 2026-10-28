# Error analysis

Error analysis means inspecting **how** a trained model fails, not only summarizing it with one metric.

A metric such as F1 tells us whether performance improved. Error analysis helps answer what should change next.

## Why it comes before changing the model

After EXP-004, the classifier still misses many real defects when the threshold is chosen to keep false alarms reasonable.

Several different causes could produce that behavior:

- defects may be too small for the current feature extraction;
- clean textures may resemble defects and produce false positives;
- the model scores for clean and defective images may overlap strongly;
- the architecture may discard useful spatial information;
- preprocessing or image resolution may hide weak defect signals.

These explanations lead to different model changes. We should therefore measure them before modifying the architecture.

## Current validation analysis

The project uses the 20-epoch weighted checkpoint and the current `0.80` operating threshold.

For every validation image we record:

- image path;
- ground-truth label;
- model defect score;
- predicted class;
- TP, TN, FP or FN category;
- ground-truth mask area as a fraction of the image.

The test split remains untouched.

## Score distributions

Plotting the scores for true clean and defective images shows how well the model separates the classes.

Ideal separation would place most clean scores low and most defect scores high, with little overlap.

Large overlap means threshold tuning alone cannot produce both high recall and few false positives.

## False negatives

False negatives are real defects predicted as clean.

For this project, they are especially important because they correspond to defects that would pass inspection.

We inspect the most confident false negatives first: defective images that receive the lowest defect scores.

Their ground-truth mask contour is shown on the image so we can see the location and scale of the missed defect.

## False positives

False positives are clean images predicted as defective.

The most confident false positives are useful because they reveal image structures that the CNN may be confusing with defects.

## Defect area: TP versus FN

The segmentation masks let us measure the real defect area even though M2 is a classification task.

For each positive image:

```math
\mathrm{defect\ area\ ratio}
=
\frac{\mathrm{defect\ pixels}}{\mathrm{all\ image\ pixels}}
```

We compare this quantity between:

- TP — defects the classifier detected;
- FN — defects the classifier missed.

If FN defects are systematically smaller, that is evidence that small defect signals are harder for the current classifier.

It would then justify investigating changes that preserve more spatial detail. We should not make that conclusion before seeing the data.

## Connection to the current CNN

The baseline progressively reduces spatial resolution with three max-pooling layers and then applies Global Average Pooling.

That design is intentionally simple and appropriate for a first baseline, but it can potentially weaken very localized signals.

Error analysis tells us whether the observed failures are actually consistent with that limitation.

## Interview explanation

A concise explanation is:

> I do not change the architecture only because a metric is unsatisfactory. I first inspect false positives and false negatives, score overlap, and relevant sample properties such as defect size. That lets me connect the observed failure mode to a specific model or data change and keep experiments controlled.
