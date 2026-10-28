# EXP-006 — Global max pooling for localized defect evidence

## Question

Does replacing global average pooling (GAP) with global max pooling (GMP) improve classification of localized defects while keeping the rest of the M2 baseline unchanged?

## Setup

Same setup as the 20-epoch weighted baseline:

- same train/validation split;
- same CNN feature extractor;
- same image size;
- same seed (`42`);
- same optimizer (Adam);
- same learning rate (`1e-3`);
- same batch size (`32`);
- same `pos_weight=8.47`;
- same 20 epochs.

Only the global aggregation changed:

- GAP baseline: average each final feature map over all spatial positions;
- GMP experiment: take the maximum activation from each final feature map.

The number of trainable parameters is unchanged.

## Training result

The best validation-loss checkpoint occurred at epoch 20:

- validation loss: `0.6978`;
- threshold `0.50`: precision `0.5323`, recall `0.6735`, F1 `0.5946`;
- TP `33`, FP `29`, FN `16`.

As in previous experiments, the checkpoint is selected by validation loss, not by F1.

## Threshold sweep

The highest tested validation F1 for the GMP checkpoint occurred at threshold `0.90`:

- precision: `0.9643`;
- recall: `0.5510`;
- F1: `0.7013`;
- TP: `27`;
- FP: `1`;
- FN: `22`.

The previous GAP checkpoint reached its highest tested validation F1 at threshold `0.80`:

- precision: `0.8485`;
- recall: `0.5714`;
- F1: `0.6829`;
- TP: `28`;
- FP: `5`;
- FN: `21`.

The best-F1 comparison therefore shows a modest F1 increase with GMP, driven mostly by much higher precision rather than higher recall.

## Comparison at similar recall

Comparing operating points with the same recall is more informative than comparing thresholds directly, because the two models produce differently distributed scores.

At recall `0.6735`:

- GAP at threshold `0.45`: precision `0.4459`, TP `33`, FP `41`, FN `16`;
- GMP at threshold `0.55`: precision `0.6226`, TP `33`, FP `20`, FN `16`.

The two models detect exactly the same number of validation defects at these operating points, but GMP reduces false positives from `41` to `20`.

At high recall, GMP also shows a better precision/recall trade-off in the tested sweep. For example:

- GAP at threshold `0.20`: recall `0.8980`, precision `0.1472`;
- GMP at threshold `0.20`: recall `0.9388`, precision `0.2054`.

## Interpretation

The original hypothesis was that GAP could dilute localized evidence by averaging each feature map over all spatial positions. GMP preserves the strongest spatial response instead.

The experiment does not show a dramatic recall increase at the best-F1 operating point, but it does show a better separation between clean and defective samples across several parts of the precision-recall trade-off.

This supports the idea that strong local activations contain useful information that GAP was partially diluting.

The expected GMP risk — reacting strongly to isolated clean-texture peaks — is still relevant, but the observed validation results do not show an overall false-positive explosion. At matched recall, GMP actually produces substantially fewer false positives than GAP.

## Conclusion

GMP is a useful improvement over pure GAP for this baseline, but it also discards information about the average/global activation pattern.

The next controlled hypothesis is to concatenate both summaries:

- global average features: `64` values;
- global max features: `64` values;
- concatenate them into `128` values;
- classify with `Linear(128, 1)`.

This tests whether global context and localized peak evidence are complementary while keeping the architecture simple.
