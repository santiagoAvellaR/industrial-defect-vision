# Experiment EXP-012 — Positive-mask Dice with lower learning rate

## Question

Does reducing the Adam learning rate from `1e-3` to `3e-4` make SimpleUNet training more stable while preserving or improving segmentation quality?

## Hypothesis

EXP-011 improved the loss formulation but still showed large epoch-to-epoch swings in precision, recall and false-positive pixels. A lower learning rate may reduce overly aggressive parameter updates and produce a more stable optimization trajectory.

## Setup

Keep EXP-011 unchanged except for the learning rate:

- Dataset: KSDD2
- Split: same train/validation split
- Seed: 42
- Model: SimpleUNet
- Augmentation: none
- Optimizer: Adam
- LR: `3e-4`
- Batch size: 4
- Loss: BCE on all images + soft Dice only on non-empty target masks
- Epochs: 10
- Hardware: CPU
- Evaluation threshold during training: 0.5
- Checkpoints: lowest validation loss and highest validation Dice

### Controlled variable

EXP-011:

`learning_rate = 1e-3`

EXP-012:

`learning_rate = 3e-4`

No other training variable changed.

## Results

The best validation-loss and best validation-Dice checkpoints both occurred at epoch 10.

| Metric | Validation at threshold 0.5 |
|---|---:|
| Loss | 0.2002 |
| Dice | 0.3897 |
| IoU | 0.2420 |
| Precision | 0.5611 |
| Recall | 0.2985 |
| TP | 64,401 |
| FP | 50,370 |
| FN | 151,368 |

Compared with EXP-011 at threshold `0.5`, Dice increased from `0.3789` to `0.3897` and IoU from `0.2337` to `0.2420`. Recall stayed nearly unchanged, while false-positive pixels decreased from `58,507` to `50,370`.

The later epochs were also more stable overall than EXP-011. Large oscillations remained, but the run no longer showed failures as extreme as the EXP-011 epoch with millions of false-positive pixels.

## Validation threshold sweep

The epoch-10 checkpoint was evaluated from threshold `0.10` to `0.90` using validation data only.

At threshold `0.50`:

- Dice: `0.3897`
- IoU: `0.2420`
- precision: `0.5611`
- recall: `0.2985`

The best validation Dice occurred at threshold `0.80`:

- Dice: `0.3942`
- IoU: `0.2455`
- precision: `0.6720`
- recall: `0.2789`

The curve had a clear local maximum around `0.75–0.80` and worsened again at higher thresholds, unlike EXP-011 where the best value occurred at the upper sweep boundary.

## Interpretation

The lower learning rate improved the model mainly by reducing false positives rather than substantially increasing true-positive coverage. This supports the hypothesis that `1e-3` was contributing to unstable optimization.

Epoch 10 was still the best epoch for both validation loss and Dice, suggesting the run may have stopped before training fully converged.

## Conclusion

EXP-012 supports keeping the positive-mask Dice formulation and using `3e-4` instead of `1e-3` for the SimpleUNet baseline. The improvement is modest but consistent with better optimization stability and fewer false-positive pixels.

The validation-selected operating threshold for this checkpoint is `0.80`.

## Next action

Run a controlled follow-up with the same model, loss, split, seed, batch size and learning rate for 20 epochs. This tests whether the 10-epoch run was truncated before convergence.

## MLflow run

Not tracked yet. MLflow will be introduced after the segmentation baseline is sufficiently stable for meaningful experiment comparison.

## Relevant commit

To be linked after EXP-012/013 documentation is finalized.
