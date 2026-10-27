# Experiment EXP-003 — Positive-class weighting for 20 epochs

## Question

Was the weighted CNN baseline in EXP-002 still undertrained at 10 epochs, and does extending the same setup to 20 epochs materially improve defect detection?

## Hypothesis

EXP-002 was still reducing both training and validation loss at epoch 10. Keeping every other setting fixed and doubling the training duration should reveal whether more optimization improves validation performance before changing architecture, sampling or learning rate.

## Setup

- Dataset: KSDD2
- Task: binary image classification (`0 = clean`, `1 = defective`)
- Train split: 1,865 images — 1,668 clean, 197 defective
- Validation split: 466 images — 417 clean, 49 defective
- Seed: 42
- Model: `BaselineCNN`
- Resolution: `640 × 224`
- Augmentation: none
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: 32
- Loss: `BCEWithLogitsLoss(pos_weight=8.47)`
- Epochs: 20
- Hardware: CPU
- Classification threshold: logit `>= 0`, equivalent to sigmoid probability `>= 0.5`
- Checkpoint criterion: lowest validation loss
- Checkpoint: `artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt`

The only intended change relative to EXP-002 was extending training from 10 to 20 epochs.

## Results

The lowest validation loss was reached at epoch 20:

| Metric | Train, epoch 20 | Validation, epoch 20 |
|---|---:|---:|
| Loss | 0.6859 | 0.7903 |
| Accuracy | 0.9078 | 0.8991 |
| Precision | 0.5483 | 0.5161 |
| Recall | 0.7208 | 0.6531 |
| F1 | 0.6228 | 0.5766 |
| TP | 142 | 32 |
| TN | 1551 | 387 |
| FP | 117 | 30 |
| FN | 55 | 17 |

However, the highest validation F1 was reached earlier, at epoch 16:

| Metric | Validation, epoch 16 |
|---|---:|
| Loss | 0.8567 |
| Accuracy | 0.9399 |
| Precision | 0.8000 |
| Recall | 0.5714 |
| F1 | 0.6667 |
| TP | 28 |
| TN | 410 |
| FP | 7 |
| FN | 21 |

The checkpoint file was **not** saved at epoch 16 because validation loss at that epoch was worse than the best validation loss already observed. The saved checkpoint at the end of the run therefore contains epoch-20 parameters, because epoch 20 had the lowest validation loss.

## Training dynamics

After epoch 10, validation loss continued to improve overall, but thresholded classification metrics remained noisy:

- epoch 11: F1 `0.6512`, recall `0.5714`;
- epoch 12: F1 `0.5825`, recall `0.6122`;
- epoch 14: F1 `0.6237`, recall `0.5918`;
- epoch 15: F1 `0.4111`, recall `0.7551`;
- epoch 16: F1 `0.6667`, recall `0.5714`;
- epoch 17: F1 `0.5312`, recall `0.6939`;
- epoch 18: F1 `0.5981`, recall `0.6531`;
- epoch 19: F1 `0.4932`, recall `0.7347`;
- epoch 20: F1 `0.5766`, recall `0.6531`.

The model clearly continues learning useful positive-class signal, but the fixed `0.5` threshold produces substantial precision/recall movement between epochs.

## Interpretation

Training for longer did reduce validation loss from `0.8648` at epoch 10 to `0.7903` at epoch 20 and increased recall at the saved checkpoint from `0.5510` to `0.6531`.

However, longer training did not produce a monotonic improvement in F1. The best F1 observed across the run was `0.6667` at epoch 16, only modestly above the `0.6506` reached at epoch 10.

This exposes an important model-selection issue:

- validation loss is continuous and independent of a classification threshold;
- F1, precision and recall depend on the chosen threshold;
- the lowest-loss epoch is not necessarily the best epoch for an operational metric.

The validation set contains only 49 positive samples, so a change of a few defect predictions can noticeably move recall and F1.

## Conclusion

The 10-epoch run was not fully converged in terms of weighted validation loss, but simply training longer does not solve the remaining decision problem.

The weighted CNN now detects meaningful defect signal. The next useful question is no longer only "how many more epochs?" but "what classification threshold gives the desired precision/recall trade-off for this inspection task?"

## Next action

Use the saved weighted checkpoint to inspect validation logits/probabilities over a range of thresholds rather than assuming `0.5` is optimal.

For each threshold, compute at least:

- TP, TN, FP, FN;
- precision;
- recall;
- F1.

This should show explicitly how lowering the threshold increases defect recall while usually increasing false positives, and how raising it does the opposite.

A later refinement may save separate checkpoints for different model-selection criteria, for example best validation loss and best validation F1, but this should be introduced only after understanding threshold selection and its implications.
