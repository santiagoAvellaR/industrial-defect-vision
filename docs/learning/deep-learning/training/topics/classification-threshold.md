# Classification threshold

A binary classifier does not directly output the final `clean` / `defective` decision.

Our CNN first produces a logit. Applying sigmoid converts that logit into a value between 0 and 1:

```math
p = \sigma(z)
```

We interpret `p` as the model score for the positive `defective` class.

The **classification threshold** is the rule that converts that continuous score into a hard class decision.

With threshold `0.5`:

```text
p >= 0.5  -> defective
p <  0.5  -> clean
```

The value `0.5` is a decision rule. It is not a learned parameter of the network and it is not inherently optimal for the industrial task.

## Moving the threshold

Lowering the threshold makes it easier to predict `defective`.

Typical effect:

```text
threshold down
-> more predicted positives
-> TP can increase
-> FN can decrease
-> recall can increase

but also
-> FP can increase
-> precision can decrease
```

Raising the threshold usually produces the opposite trade-off.

This is why threshold selection depends on the cost of mistakes. In industrial inspection, missing a real defect (`FN`) may be more costly than generating an extra false alarm (`FP`).

## No retraining is required

Changing the threshold does not change the CNN weights.

The model can produce the same scores once, and those scores can then be evaluated with many decision thresholds.

This separates two problems:

1. **learning** — produce useful scores from images;
2. **decision policy** — decide which score is high enough to call a sample defective.

## Validation, not test, chooses the threshold

The threshold is a model-selection decision. Therefore it must be studied and selected on validation data.

The test split remains untouched while choosing the threshold. It is reserved for the final evaluation after model and decision-policy choices have been made.

## Threshold sweep in M2

The project evaluates thresholds from `0.10` to `0.90` using:

```bash
python scripts/evaluate_thresholds.py \
  data/raw/ksdd2 \
  artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt
```

For each threshold we inspect:

- precision;
- recall;
- F1;
- TP;
- FP;
- FN.

The generated figure `artifacts/figures/threshold_sweep.png` shows how precision, recall and F1 change as the decision threshold moves.

For EXP-004, the highest F1 in the tested grid occurred at threshold `0.80`:

- precision `0.8485`;
- recall `0.5714`;
- F1 `0.6829`;
- TP `28`;
- FP `5`;
- FN `21`.

This is the best F1 point on that validation sweep, not automatically the final industrial operating threshold.

## Scores are not automatically calibrated probabilities

The current model was trained with `BCEWithLogitsLoss(pos_weight=8.47)`.

Positive-class weighting changes the optimization objective and can shift model scores. Therefore a sigmoid output between `0` and `1` should be treated here primarily as a **defect score used for ranking and decisions**, unless probability calibration is explicitly evaluated.

In other words, a score such as `0.80` does not automatically mean "an 80% real-world probability of defect".

This also helps explain why a useful decision threshold can be far from `0.5`.

## What threshold tuning cannot fix

Threshold tuning only changes the decision boundary over scores the model already produces.

If high recall can only be obtained with a huge number of false positives, the problem is not just the threshold. The model may not separate clean and defective examples well enough.

In EXP-004, obtaining recall around `0.8` required many false positives. That is evidence that the baseline representation still has limited class separation.

## Interview explanation

A concise explanation is:

> The network outputs a continuous score. The classification threshold turns that score into a binary decision. Lower thresholds usually improve recall at the cost of more false positives, while higher thresholds usually improve precision at the cost of more false negatives. I select the threshold on validation data according to the operational cost of those errors, not automatically at 0.5. I also avoid interpreting weighted-loss sigmoid outputs as calibrated probabilities unless calibration has been checked.
