# Experiment EXP-004 — Validation threshold sweep

## Question

How does the classification threshold change the precision-recall trade-off of the 20-epoch weighted CNN checkpoint, and which thresholds are useful candidates for the industrial defect-detection task?

## Setup

- Dataset: KSDD2
- Split used for threshold selection: validation only
- Validation samples: 466 — 417 clean, 49 defective
- Model checkpoint: `artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt`
- Checkpoint epoch: 20
- Checkpoint validation loss: `0.7903`
- Training loss: `BCEWithLogitsLoss(pos_weight=8.47)`
- Thresholds tested: `0.10` to `0.90` in steps of `0.05`
- No retraining was performed
- Test split remains untouched

## Results

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.1052 | 1.0000 | 0.1903 | 49 | 417 | 0 |
| 0.15 | 0.1052 | 1.0000 | 0.1903 | 49 | 417 | 0 |
| 0.20 | 0.1472 | 0.8980 | 0.2529 | 44 | 255 | 5 |
| 0.25 | 0.1818 | 0.8163 | 0.2974 | 40 | 180 | 9 |
| 0.30 | 0.2364 | 0.7959 | 0.3645 | 39 | 126 | 10 |
| 0.35 | 0.2992 | 0.7755 | 0.4318 | 38 | 89 | 11 |
| 0.40 | 0.3617 | 0.6939 | 0.4755 | 34 | 60 | 15 |
| 0.45 | 0.4459 | 0.6735 | 0.5366 | 33 | 41 | 16 |
| 0.50 | 0.5161 | 0.6531 | 0.5766 | 32 | 30 | 17 |
| 0.55 | 0.5636 | 0.6327 | 0.5962 | 31 | 24 | 18 |
| 0.60 | 0.6250 | 0.6122 | 0.6186 | 30 | 18 | 19 |
| 0.65 | 0.6744 | 0.5918 | 0.6304 | 29 | 14 | 20 |
| 0.70 | 0.8000 | 0.5714 | 0.6667 | 28 | 7 | 21 |
| 0.75 | 0.8235 | 0.5714 | 0.6747 | 28 | 6 | 21 |
| 0.80 | 0.8485 | 0.5714 | 0.6829 | 28 | 5 | 21 |
| 0.85 | 0.8387 | 0.5306 | 0.6500 | 26 | 5 | 23 |
| 0.90 | 0.8846 | 0.4694 | 0.6133 | 23 | 3 | 26 |

The highest validation F1 in the tested grid is at threshold `0.80`:

- precision: `0.8485`;
- recall: `0.5714`;
- F1: `0.6829`;
- TP: `28`;
- FP: `5`;
- FN: `21`.

## Interpretation

The sweep clearly shows the expected precision-recall trade-off.

Low thresholds detect almost every real defect, but generate many false alarms. For example, threshold `0.20` reaches recall `0.8980`, but creates `255` false positives.

Higher thresholds strongly reduce false alarms, but eventually start missing more defects. Threshold `0.90` produces only `3` false positives, but recall falls to `0.4694`.

Threshold `0.80` is the best F1 point on this discrete validation sweep, but that does not make it automatically the correct industrial operating point. F1 weights precision and recall symmetrically, while the real system may assign a much higher cost to false negatives than to false positives.

An interesting local result is that thresholds `0.70`, `0.75` and `0.80` all detect the same `28 / 49` defects. Increasing the threshold within that interval only removes false positives (`7 -> 6 -> 5`), so `0.80` dominates the other two on this validation set.

## Important calibration note

The model was trained with `pos_weight=8.47`. Positive-class weighting changes the optimization objective and can shift the model scores. Therefore sigmoid outputs should be treated here primarily as ranking / decision scores unless probability calibration is explicitly evaluated.

This is another reason not to assume that threshold `0.5` is special or automatically optimal.

## Conclusion

Threshold selection is a separate decision-policy problem from model training.

The current checkpoint can move from very high recall with many false alarms to high precision with lower recall without changing a single model weight.

For this baseline, the threshold sweep also exposes a model limitation: achieving recall above roughly `0.8` requires an unacceptably large number of false positives. This suggests that threshold tuning alone cannot solve the task; improving the learned representation/model will eventually be necessary.

## Next action

Do not select the final operating threshold yet.

Use this experiment to establish the precision-recall behavior of the M2 baseline, keep the test set untouched, and continue improving the model. Later, select an operating threshold according to an explicit industrial requirement such as a minimum acceptable recall or a cost ratio between false negatives and false positives.

## Relevant artifacts

- `scripts/evaluate_thresholds.py`
- `artifacts/figures/threshold_sweep.png`
- `docs/learning/deep-learning/training/topics/classification-threshold.md`
