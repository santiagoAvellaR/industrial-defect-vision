# Experiment EXP-005 — Classification error analysis

## Question

What failure modes explain the remaining validation errors of the 20-epoch weighted CNN baseline?

## Setup

- Dataset: KSDD2 validation split
- Checkpoint: `artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt`
- Checkpoint epoch: 20
- Positive-class weight: `8.47`
- Global pooling: average
- Operating threshold: `0.80`
- Validation samples: 466
- Defective validation samples: 49

The test split remains untouched.

## Results at threshold 0.80

| Metric | Value |
|---|---:|
| TP | 28 |
| TN | 412 |
| FP | 5 |
| FN | 21 |
| Precision | 0.8485 |
| Recall | 0.5714 |
| F1 | 0.6829 |

The median ground-truth defect area differs strongly between detected and missed positives:

- TP median defect area: `1.852%` of the image;
- FN median defect area: `0.679%` of the image.

Fifteen of the 21 false negatives have defect areas below the TP median, so small/localized defects are clearly harder for the current classifier.

However, defect size does not explain all failures. Some false negatives contain large ground-truth regions, including one near `29.8%` of the image. The relationship between defect area and model score is therefore weak rather than deterministic.

## Visual observations

The score-distribution plot shows useful class separation but also substantial overlap. Most clean samples receive relatively low scores, while many defective samples cluster near very high scores, but a non-trivial defective group remains in the low-score region.

The false-negative contact sheet contains many small or highly localized defects, including defects near image boundaries, but also several visually large regions. This means the current failure mode is broader than a simple minimum-size limitation.

The false-positive examples are visually close to normal surface texture. The model therefore sometimes assigns high defect scores to normal local texture variations.

## Interpretation

The current model is learning useful defect evidence, but its image-level aggregation appears to be a plausible bottleneck.

After three spatial max-pooling operations, the final feature tensor has shape `[B, 64, 80, 28]`. Global Average Pooling then averages each feature map across all `80 × 28 = 2240` spatial positions. A strong activation localized to a small region can therefore be diluted by the many locations without that evidence.

This hypothesis is consistent with the smaller median area of false negatives, but it remains only a hypothesis because large false negatives also exist.

## Next controlled experiment

Change only the final global aggregation from Global Average Pooling to Global Max Pooling.

Global Max Pooling asks whether a strong learned feature appeared anywhere in the image instead of averaging that feature over the full spatial field.

Keep unchanged:

- convolutional backbone;
- parameter count;
- input resolution;
- train/validation split;
- seed;
- optimizer and learning rate;
- `pos_weight=8.47`;
- batch size;
- 20 training epochs.

The main risk is increased sensitivity to isolated texture/noise, which could improve recall while increasing false positives. The experiment should therefore be judged with precision, recall, F1, confusion counts and a new threshold sweep rather than accuracy alone.
