# Experiment EXP-011 — BCE + Dice on positive masks only

## Question

Does excluding empty target masks from the Dice term reduce the tendency toward all-background predictions and improve foreground recall and Dice?

## Hypothesis

BCE should continue supervising every pixel in every image, including clean images. Restricting the overlap-based Dice term to masks that actually contain foreground may reduce unnecessary pressure toward zero foreground probabilities on the many clean KSDD2 images.

Expected effect:

- validation recall should increase;
- Dice and IoU should improve if the hypothesis is correct;
- precision may decrease somewhat as the model becomes less conservative;
- epoch-to-epoch all-background collapses should become less frequent or disappear.

## Setup

Keep the EXP-010 setup unchanged except for the Dice scope:

- Dataset: KSDD2
- Split: same train/validation split as EXP-010
- Seed: 42
- Model: SimpleUNet
- Resolution: dataset default preprocessing
- Augmentation: none
- Optimizer: Adam
- LR: `1e-3`
- Batch size: 4
- Loss: BCE on all images + soft Dice only on non-empty target masks
- Epochs: 10
- Hardware: CPU
- Evaluation threshold: 0.5
- Checkpoints: lowest validation loss and highest validation Dice

### Controlled variable

EXP-010:

`BCE(all images) + Dice(all images)`

EXP-011:

`BCE(all images) + Dice(positive-mask images only)`

No other training variable should change.

## Results

To be filled after training.

| Metric | Best validation loss checkpoint | Best validation Dice checkpoint |
|---|---:|---:|
| Loss | | |
| Dice | | |
| IoU | | |
| Precision | | |
| Recall | | |

## Qualitative observations

To be filled after training and prediction inspection.

## Failure modes

To be filled after training.

## Conclusion

Pending experiment results.

## Next action

Compare EXP-011 directly with EXP-010 before changing the learning rate, adding weighted BCE, introducing augmentation or changing the architecture.

## MLflow run

Not tracked yet.

## Relevant commit

To be linked after EXP-011 is finalized.
