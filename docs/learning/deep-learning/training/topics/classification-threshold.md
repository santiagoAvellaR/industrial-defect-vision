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

The model can produce the same probabilities once, and those probabilities can then be evaluated with many decision thresholds.

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

## Interview explanation

A concise explanation is:

> The network outputs a continuous score. The classification threshold turns that score into a binary decision. Lower thresholds usually improve recall at the cost of more false positives, while higher thresholds usually improve precision at the cost of more false negatives. I select the threshold on validation data according to the operational cost of those errors, not automatically at 0.5.
