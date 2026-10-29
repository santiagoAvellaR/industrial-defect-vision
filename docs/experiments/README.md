# Experiments

Human-readable summaries of relevant experiments belong here.

MLflow will later store machine-readable runs; these documents capture the question, setup, conclusion and next action.

## M2 classification experiments

1. [EXP-001 — Unweighted CNN classification baseline](EXP-001-unweighted-cnn-baseline.md) — majority-class collapse with zero defect recall.
2. [EXP-002 — CNN baseline with positive-class weighting](EXP-002-pos-weight-cnn-baseline.md) — `pos_weight=8.47` breaks the collapse and produces useful defect detection.
3. [EXP-003 — Positive-class weighting for 20 epochs](EXP-003-pos-weight-20-epochs.md) — longer training lowers validation loss but reveals a mismatch between best-loss and best-F1 epochs.
4. [EXP-004 — Validation threshold sweep](EXP-004-validation-threshold-sweep.md) — threshold tuning exposes the precision-recall trade-off without retraining; best tested validation F1 occurs at `0.80`.
5. [EXP-005 — Classification error analysis](EXP-005-classification-error-analysis.md) — missed defects are typically smaller, but size alone does not explain the errors; global aggregation is the next controlled hypothesis.
6. [EXP-006 — Global max pooling](EXP-006-global-max-pooling.md) — GMP improves the precision-recall trade-off over GAP, especially when comparing operating points at matched recall.
7. [EXP-007 — Combined global average and max pooling](EXP-007-combined-global-pooling.md) — GAP+GMP improves several matched-recall operating points, but the best tested F1 remains `0.7013`, supporting closure of the scratch-CNN baseline phase.

## M3 transfer-learning experiments

8. [EXP-008 — Frozen pretrained ResNet18](EXP-008-frozen-resnet18.md) — pretrained features alone are useful but underperform the scratch CNN; the result motivates partial fine-tuning to adapt to the industrial domain.
