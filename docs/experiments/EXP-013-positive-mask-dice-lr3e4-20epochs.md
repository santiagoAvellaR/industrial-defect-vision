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
- Hardware: to be recorded after run
- Evaluation threshold during training: 0.5
- Checkpoints: lowest validation loss and highest validation Dice

### Controlled variable

EXP-012:

`epochs = 10`

EXP-013:

`epochs = 20`

No other training variable should change.

## Results

Pending.

## Questions to answer after the run

1. Does best validation Dice improve beyond EXP-012 (`0.3897` at threshold 0.5)?
2. Does validation loss continue decreasing after epoch 10?
3. Do metrics plateau, continue improving, or begin to overfit?
4. Does the lower learning rate remain reasonably stable over the longer run?
5. Which epoch is selected by best validation Dice and best validation loss?

## Next action

After training, run a threshold sweep on the best-Dice checkpoint before deciding whether to change another training variable.

## MLflow run

Not tracked yet.
