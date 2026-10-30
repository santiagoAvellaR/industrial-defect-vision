# Experiment EXP-010 — SimpleUNet with BCE + Dice

## Question

Can a small U-Net learn meaningful KSDD2 defect segmentation using a simple combined BCE + soft Dice objective?

## Hypothesis

BCE should provide local pixel-wise supervision while Dice should reduce the dominance of the large background class and encourage foreground overlap.

## Setup

- Dataset: KSDD2
- Split: existing reproducible train/validation split
- Seed: 42
- Model: SimpleUNet
- Resolution: dataset default preprocessing
- Augmentation: none
- Optimizer: Adam
- LR: `1e-3`
- Batch size: 4
- Loss: `BCEWithLogitsLoss + soft Dice loss`
- Epochs: 10
- Hardware: CPU
- Evaluation threshold: 0.5
- Checkpoint selection: lowest validation loss

## Results

Best validation loss was reached at epoch 3:

| Metric | Validation |
|---|---:|
| Loss | 0.1298 |
| Dice | 0.3205 |
| IoU | 0.1908 |
| Precision | 0.5862 |
| Recall | 0.2205 |

The highest observed validation Dice during the 10-epoch run occurred at epoch 8:

| Metric | Validation |
|---|---:|
| Loss | 0.1515 |
| Dice | 0.3469 |
| IoU | 0.2098 |
| Precision | 0.5083 |
| Recall | 0.2632 |

A validation threshold sweep was then run on the saved epoch-3 checkpoint from `0.10` to `0.90` in steps of `0.05`.

| Threshold | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.10 | 0.3243 | 0.1935 | 0.5735 | 0.2260 |
| 0.50 | 0.3205 | 0.1908 | 0.5862 | 0.2205 |
| 0.90 | 0.3087 | 0.1825 | 0.6054 | 0.2072 |

The best validation Dice in the sweep was `0.3243` at threshold `0.10`, only `0.0038` above the Dice at threshold `0.50`.

## Qualitative observations

The model learned non-trivial foreground localization, but several epochs produced no positive pixels at the fixed 0.5 threshold. Other epochs recovered useful segmentation.

The threshold sweep produced nearly flat metric curves. Lowering the threshold from `0.50` to `0.10` did not materially recover the missed foreground pixels.

## Failure modes

- Strong epoch-to-epoch instability in hard segmentation metrics.
- Several train and validation epochs collapsed to all-background predictions at threshold 0.5.
- Checkpoint selection by validation loss did not select the epoch with the best observed Dice/IoU.
- Validation recall remained low, around `0.22` for the saved checkpoint, indicating strong under-segmentation.
- KSDD2 contains many clean images and severe pixel-level foreground/background imbalance.
- The current Dice term is calculated per image and averaged across the batch, including empty target masks.

## Conclusion

EXP-010 shows that SimpleUNet can learn meaningful defect localization, but the baseline is not yet stable enough to treat as a reliable segmentation reference.

The validation threshold sweep rules out threshold selection as the primary explanation for the low recall: Dice changes only slightly across thresholds from `0.10` to `0.90`. The dominant issue is therefore in the learned segmentation behavior rather than in converting probabilities to binary masks.

The next controlled hypothesis is that applying per-image Dice to many empty masks creates additional pressure toward all-background predictions. This should be tested without changing the architecture, optimizer, learning rate, split, batch size, epoch count or evaluation threshold.

## Next action

Run EXP-011 with BCE on every image but compute the Dice term only for target masks that contain foreground. Also save checkpoints independently by lowest validation loss and highest validation Dice.

## MLflow run

Not tracked yet. MLflow will be introduced after the segmentation training pipeline is stable.

## Relevant commit

Threshold-sweep and EXP-011 preparation are implemented on the M4 segmentation feature branch.
