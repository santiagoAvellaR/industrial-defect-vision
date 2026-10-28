# EXP-007 — Combined global average and max pooling

## Question

Can the classifier benefit from keeping both global context and the strongest localized activation instead of choosing only GAP or GMP?

## Setup

The experiment keeps the same M2 classification setup used in the controlled pooling comparison:

- same train/validation split;
- seed `42`;
- `20` epochs;
- batch size `32`;
- Adam with learning rate `1e-3`;
- `BCEWithLogitsLoss(pos_weight=8.47)`;
- same three convolution blocks;
- same image preprocessing.

The only model change is the final spatial aggregation:

```text
final features [B, 64, 80, 28]

GAP -> [B, 64]
GMP -> [B, 64]

concatenate -> [B, 128]
Linear(128, 1)
```

This adds only 64 classifier weights relative to the single-pooling variants.

## Best checkpoint

The checkpoint selected by validation loss was epoch `19`:

- validation loss: `0.6419`;
- at the default threshold `0.50`: precision `0.5806`, recall `0.7347`, F1 `0.6486`;
- TP `36`, FP `26`, FN `13`.

Epoch 20 was not saved because validation loss increased to `0.6849`.

## Threshold sweep

The best tested validation F1 occurred at threshold `0.90`:

- precision: `0.9643`;
- recall: `0.5510`;
- F1: `0.7013`;
- TP: `27`;
- FP: `1`;
- FN: `22`.

This maximum F1 is numerically identical to the GMP experiment, but comparing only the single maximum hides differences along the precision-recall trade-off.

## Matched-recall comparison

At recall `0.5714`:

| Pooling | Threshold | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| GAP | 0.80 | 0.8485 | 28 | 5 | 21 |
| GMP | 0.75 | 0.8000 | 28 | 7 | 21 |
| GAP + GMP | 0.80 | 0.8750 | 28 | 4 | 21 |

At recall `0.6122`:

| Pooling | Threshold | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| GMP | 0.60 | 0.6383 | 30 | 17 | 19 |
| GAP + GMP | 0.70 | 0.7692 | 30 | 9 | 19 |

The combined representation therefore improves several operating points even though its maximum tested F1 is the same as GMP.

## Interpretation

GAP and GMP contain complementary summaries:

- GAP represents how strongly a feature is present across the image overall;
- GMP represents whether a very strong localized activation exists somewhere;
- concatenating them lets the final linear layer decide how much each summary matters.

The result supports the hypothesis that both global context and localized evidence are useful for this dataset.

However, the improvement over GMP is modest and the best validation recall at a high-precision operating point remains around `55–60%`. Continuing to add handcrafted pooling variants would therefore have diminishing value.

## Conclusion

EXP-007 is a useful final refinement of the scratch CNN baseline, but not a reason to keep expanding M2.

The next meaningful comparison is transfer learning: use a pretrained visual backbone and test whether stronger learned features improve separation of clean and defective samples, especially for difficult localized defects.
