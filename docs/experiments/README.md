# Experiments

Human-readable summaries of relevant experiments belong here.

MLflow will later store machine-readable runs; these documents capture the question, setup, conclusion and next action.

## M2 classification experiments

1. [EXP-001 — Unweighted CNN classification baseline](EXP-001-unweighted-cnn-baseline.md) — majority-class collapse with zero defect recall.
2. [EXP-002 — CNN baseline with positive-class weighting](EXP-002-pos-weight-cnn-baseline.md) — `pos_weight=8.47` breaks the collapse and produces useful defect detection.
3. [EXP-003 — Positive-class weighting for 20 epochs](EXP-003-pos-weight-20-epochs.md) — longer training lowers validation loss but reveals a mismatch between best-loss and best-F1 epochs.
4. [EXP-004 — Validation threshold sweep](EXP-004-validation-threshold-sweep.md) — threshold tuning exposes the precision-recall trade-off without retraining; best tested validation F1 occurs at `0.80`.
5. [EXP-005 — Classification error analysis](EXP-005-classification-error-analysis.md) — missed defects are typically smaller, but size alone does not explain the errors; global aggregation is the next controlled hypothesis.
