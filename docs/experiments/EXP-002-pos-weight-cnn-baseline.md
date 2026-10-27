# Experiment EXP-002 — CNN baseline with positive-class weighting

## Question

Does adding positive-class weighting to `BCEWithLogitsLoss` prevent the CNN baseline from collapsing to the majority `clean` class?

## Hypothesis

The unweighted baseline in EXP-001 predicted every image as clean because defective samples are strongly underrepresented.

Using a positive-class weight equal to the training-set negative-to-positive ratio should increase the contribution of defective samples to the loss and produce non-zero defect recall without changing the model or sampling strategy.

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
- Epochs: 10
- Hardware: CPU
- Classification threshold: logit `>= 0`, equivalent to sigmoid probability `>= 0.5`
- Checkpoint: `artifacts/checkpoints/baseline_cnn_pos_weight.pt`

The only intended change relative to EXP-001 was the positive-class weight.

## Results

The best validation loss was reached at epoch 10.

| Metric | Train, epoch 10 | Validation, epoch 10 |
|---|---:|---:|
| Loss | 0.8026 | 0.8648 |
| Accuracy | 0.9260 | 0.9378 |
| Precision | 0.6513 | 0.7941 |
| Recall | 0.6447 | 0.5510 |
| F1 | 0.6480 | 0.6506 |
| TP | 127 | 27 |
| TN | 1600 | 410 |
| FP | 68 | 7 |
| FN | 70 | 22 |

Validation behavior changed substantially compared with EXP-001:

| Metric | EXP-001 | EXP-002 |
|---|---:|---:|
| Accuracy | 0.8948 | 0.9378 |
| Precision | 0.0000 | 0.7941 |
| Recall | 0.0000 | 0.5510 |
| F1 | 0.0000 | 0.6506 |
| TP | 0 | 27 |
| TN | 417 | 410 |
| FP | 0 | 7 |
| FN | 49 | 22 |

The weighted loss values are not directly comparable with the unweighted loss values from EXP-001 because the objective itself changed.

## Training dynamics

The first epochs were unstable at the fixed `0.5` classification threshold:

- epoch 1 validation: recall `0.8776`, precision `0.1049`;
- epoch 2 validation: recall `0.0204`, precision `1.0000`;
- epoch 3 validation: recall `1.0000`, precision `0.1052`.

The model therefore moved between predicting many positives and almost no positives early in training.

From roughly epoch 7 onward, the behavior became more balanced and the validation F1 improved:

- epoch 7: F1 `0.5600`;
- epoch 8: F1 `0.5143`;
- epoch 9: F1 `0.5652`;
- epoch 10: F1 `0.6506`.

Training and validation losses were still decreasing at epoch 10, so there is no evidence yet that the run had clearly converged.

## Interpretation

Positive-class weighting solved the specific failure mode observed in EXP-001: the model no longer predicts every image as clean.

At epoch 10 the classifier detects `27 / 49` validation defects and produces only 7 false alarms. This is a major qualitative improvement over the majority-class collapse.

However, recall is still only `0.5510`, meaning `22 / 49` real defects are missed. For an industrial inspection system, that false-negative rate is still too high to consider the classifier satisfactory.

The similar train and validation F1 values at epoch 10 (`0.6480` vs `0.6506`) do not suggest obvious overfitting at this point, although the validation metrics are noisy because the positive validation set contains only 49 samples.

## Conclusion

`pos_weight=8.47` is effective at forcing the baseline to learn useful positive-class signal. The experiment validates class weighting as a necessary improvement over the unweighted baseline.

The model is now learning the actual task rather than only the class prior, but training is not yet mature and recall remains insufficient.

## Next action

Before adding another imbalance mechanism such as weighted sampling, keep the current setup and investigate whether the model is still undertrained.

A controlled next experiment should extend the number of epochs while keeping model, split, seed, optimizer, learning rate, batch size and `pos_weight` unchanged. The best checkpoint should continue to be selected on validation performance.

After obtaining a better-converged model, inspect the validation prediction scores and study the decision threshold explicitly. The fixed `0.5` threshold is a decision rule, not a property of the network, and changing it trades precision against recall without retraining.

## Relevant implementation

- `src/industrial_vision/models/baseline_cnn.py`
- `src/industrial_vision/training/classification.py`
- `scripts/train_baseline.py`
