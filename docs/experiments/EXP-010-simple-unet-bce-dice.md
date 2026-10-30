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

## Qualitative observations

The model learned non-trivial foreground localization, but several epochs produced no positive pixels at the fixed 0.5 threshold. Other epochs recovered useful segmentation.

## Failure modes

- Strong epoch-to-epoch instability in hard segmentation metrics.
- Several train and validation epochs collapsed to all-background predictions at threshold 0.5.
- Checkpoint selection by validation loss did not select the epoch with the best observed Dice/IoU.
- KSDD2 contains many clean images and severe pixel-level foreground/background imbalance. The current Dice term is calculated per image and then averaged, so empty target masks deserve explicit investigation.
- A fixed threshold of 0.5 may hide useful probability separation when foreground probabilities remain below 0.5.

## Conclusion

EXP-010 shows that SimpleUNet can learn meaningful defect localization, but the baseline is not yet stable enough to treat as a reliable segmentation reference. The observed failure pattern is not classic overfitting because training metrics also collapse. Before changing the optimizer, learning rate, architecture, or loss, the current checkpoint should be diagnosed with a validation threshold sweep and probability-map inspection.

## Next action

Evaluate the saved checkpoint over validation thresholds from 0.10 to 0.90 and compare Dice, IoU, precision and recall. Use the result to decide whether the dominant issue is threshold sensitivity or whether a new training experiment is needed.

## MLflow run

Not tracked yet. MLflow will be introduced after the segmentation training pipeline is stable.

## Relevant commit

To be linked after the threshold-sweep implementation is finalized.
