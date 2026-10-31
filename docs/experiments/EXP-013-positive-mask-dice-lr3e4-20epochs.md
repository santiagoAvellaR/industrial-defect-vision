# Experiment EXP-013 — Positive-mask Dice, LR 3e-4, 20 epochs

## Question

Was EXP-012 stopped before convergence because training ended at epoch 10 while validation Dice and validation loss were still improving?

## Hypothesis

Keeping the EXP-012 configuration unchanged and training for 20 epochs may allow the lower-learning-rate run to continue improving. If validation metrics plateau or degrade instead, then simply training longer is not enough and the next experiment should change another component such as loss weighting, sampling or augmentation.

## Setup

Keep EXP-012 unchanged except for training duration:

- Dataset: KSDD2
- Split: same train/validation split
- Seed: 42
- Model: SimpleUNet
- Augmentation: none
- Optimizer: Adam
- LR: `3e-4`
- Batch size: 4
- Loss: BCE on all images + soft Dice only on non-empty target masks
- Epochs: 20
- Hardware: CPU
- Evaluation threshold during training: 0.5
- Checkpoints: lowest validation loss and highest validation Dice

### Controlled variable

EXP-012:

`epochs = 10`

EXP-013:

`epochs = 20`

No other training variable changed.

## Results

Training beyond epoch 10 improved the best validation result, so EXP-012 had stopped before the best observed model state.

Best validation-Dice checkpoint:

| Metric | EXP-012 best Dice | EXP-013 best Dice |
|---|---:|---:|
| Epoch | 10 | 17 |
| Dice | 0.3897 | 0.4201 |
| IoU | 0.2420 | 0.2659 |
| Precision | 0.5611 | 0.6702 |
| Recall | 0.2985 | 0.3060 |

Best validation-loss checkpoint in EXP-013 occurred at epoch 19:

- validation loss: `0.1948`
- Dice: `0.4199`
- IoU: `0.2657`
- precision: `0.6511`
- recall: `0.3099`

The best Dice improved by `+0.0304` absolute relative to EXP-012 at the common training threshold of `0.5`.

## Interpretation

The 20-epoch run shows that ten epochs were not sufficient for this lower-learning-rate configuration. Useful validation improvements occurred well after epoch 10, with the best Dice at epoch 17 and best loss at epoch 19.

However, optimization is still not completely smooth. Validation metrics continue to move between more conservative and more aggressive foreground predictions across epochs, and training metrics also oscillate. The longer run therefore improves the model but does not remove the remaining instability.

There is no clear monotonic overfitting pattern in this run: validation quality reaches new best values late in training rather than consistently degrading while training improves.

## Conclusion

EXP-013 supports the hypothesis that EXP-012 was stopped too early. Extending training from 10 to 20 epochs improved validation Dice and IoU while preserving the lower learning rate and loss formulation.

The next expensive training experiment should not be launched on the local CPU. Remote GPU execution should be introduced first so subsequent controlled experiments can be run efficiently without changing the training code.

## Next action

1. Introduce a reproducible remote-GPU workflow while keeping code in Git and datasets/checkpoints outside Git.
2. Run a validation threshold sweep on the EXP-013 best-Dice checkpoint.
3. Only after closing EXP-013 decide whether the next modeling experiment should change loss weighting, sampling, augmentation or another single variable.

## MLflow run

Not tracked yet.
