# Experiment EXP-001 — Unweighted CNN classification baseline

## Question

Can the small CNN baseline learn to distinguish defective from clean KSDD2 images when trained with plain `BCEWithLogitsLoss`, without any class-imbalance correction?

## Hypothesis

Because defective images are a small minority of the training set, an unweighted binary loss may allow the model to obtain deceptively high accuracy by predicting the majority `clean` class for every image.

The purpose of this first run is to observe that behavior before introducing class weighting or resampling.

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
- Loss: unweighted `BCEWithLogitsLoss`
- Epochs: 10
- Hardware: CPU
- Classification threshold: logit `>= 0`, equivalent to sigmoid probability `>= 0.5`

## Results

The best validation loss was reached at epoch 9.

| Metric | Train, epoch 9 | Validation, epoch 9 |
|---|---:|---:|
| Loss | 0.3369 | 0.3304 |
| Accuracy | 0.8944 | 0.8948 |
| Precision | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 |
| F1 | 0.0000 | 0.0000 |
| TP | 0 | 0 |
| TN | 1668 | 417 |
| FP | 0 | 0 |
| FN | 197 | 49 |

Validation loss improved from `0.3365` at epoch 1 to `0.3304` at epoch 9, but the classification behavior never changed: every sample was predicted as `clean`.

## Interpretation

The accuracy is not evidence of useful defect detection.

On the validation set, predicting every image as clean gives:

```math
\mathrm{accuracy}
=
\frac{417}{417+49}
\approx 0.895
```

which is essentially the observed `0.8948` accuracy.

The same happens on the training set: the reported accuracy matches the fraction of clean samples almost exactly.

This is a majority-class collapse.

Plain binary cross-entropy is optimized over all samples. Since clean images strongly outnumber defective images, the model can reduce aggregate loss substantially by assigning a low defect probability broadly, even if it never crosses the `0.5` classification threshold for positive samples.

This is why loss and accuracy alone would have been misleading. `TP`, `FN`, recall and F1 reveal the actual failure immediately.

## Failure mode

The model learned a trivial majority-class solution:

- all clean images are classified correctly;
- all defective images are missed;
- recall for the industrial defect class is zero.

For an inspection system this behavior is unacceptable because false negatives correspond to undetected defects.

## Conclusion

The unweighted baseline establishes an important reference point: the current dataset imbalance is strong enough for the CNN to collapse to the clean class under the current training setup.

The result supports introducing an explicit imbalance treatment in the next controlled experiment rather than adding it pre-emptively without evidence.

## Next action

Run a second experiment with positive-class weighting in `BCEWithLogitsLoss` while keeping the model, split, optimizer, seed, batch size and learning rate unchanged.

A natural initial weight is the training-set negative-to-positive ratio:

```math
\mathrm{pos\_weight}
=
\frac{N_{\mathrm{negative}}}{N_{\mathrm{positive}}}
=
\frac{1668}{197}
\approx 8.47
```

This isolates the effect of class weighting. Weighted sampling should not be introduced at the same time because that would change two variables in one experiment.

## Relevant implementation

- `src/industrial_vision/models/baseline_cnn.py`
- `src/industrial_vision/training/classification.py`
- `scripts/train_baseline.py`
