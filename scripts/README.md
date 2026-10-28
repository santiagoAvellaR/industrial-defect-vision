# Scripts

Executable entry points should remain thin and call reusable code from `src/industrial_vision/`.

## KSDD2 inspection

After extracting KSDD2 locally, run:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

The script checks image/mask pairing, defective versus clean samples, and image/mask size mismatches for the official `train` and `test` folders.

## Exact duplicate check

After generating `data/splits/train.csv`, `val.csv` and `test.csv`, run:

```bash
python scripts/check_duplicates.py data/raw/ksdd2
```

The script hashes the original images and reports exact duplicates. It exits with an error if the same image appears in more than one split, because that would create data leakage.

## M2 CNN baseline training

Train the first classification baseline with:

```bash
python scripts/train_baseline.py data/raw/ksdd2
```

Defaults:

- train manifest: `data/splits/train.csv`;
- validation manifest: `data/splits/val.csv`;
- epochs: `10`;
- batch size: `32`;
- learning rate: `1e-3`;
- positive-class weight: none;
- global pooling: average;
- seed: `42`;
- best checkpoint: `artifacts/checkpoints/baseline_cnn.pt`.

The script reports loss, accuracy, precision, recall, F1 and TP/TN/FP/FN for train and validation. The checkpoint with the lowest validation loss is saved.

Example with explicit settings:

```bash
python scripts/train_baseline.py data/raw/ksdd2 \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-3
```

To run the first positive-class-weighted experiment while leaving the rest of the setup unchanged:

```bash
python scripts/train_baseline.py data/raw/ksdd2 \
  --pos-weight 8.47 \
  --checkpoint artifacts/checkpoints/baseline_cnn_pos_weight.pt
```

The separate checkpoint path avoids overwriting the unweighted EXP-001 checkpoint.

To compare global max pooling against the 20-epoch weighted average-pooling baseline:

```bash
python scripts/train_baseline.py data/raw/ksdd2 \
  --epochs 20 \
  --pos-weight 8.47 \
  --global-pool max \
  --checkpoint artifacts/checkpoints/baseline_cnn_global_max_20e.pt
```

This controlled experiment changes only the final global aggregation from average to maximum. The checkpoint stores the pooling mode so evaluation scripts can reconstruct the correct model. Older checkpoints without this field are interpreted as average-pooling checkpoints.

## Validation threshold sweep

A trained checkpoint can be evaluated at different classification thresholds without retraining:

```bash
python scripts/evaluate_thresholds.py \
  data/raw/ksdd2 \
  artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt
```

The script:

- loads the saved model;
- computes validation probabilities once;
- evaluates thresholds from `0.10` to `0.90` in steps of `0.05`;
- reports precision, recall, F1, TP, FP and FN for each threshold;
- prints the threshold with the highest validation F1;
- saves `artifacts/figures/threshold_sweep.png`.

Threshold selection is performed on validation data. The untouched test split must not be used to choose the threshold.

## Classification error analysis

Inspect validation errors from the 20-epoch weighted checkpoint at the current `0.80` operating threshold:

```bash
python scripts/analyze_classification_errors.py \
  data/raw/ksdd2 \
  artifacts/checkpoints/baseline_cnn_pos_weight_20e.pt \
  --threshold 0.80
```

The script saves outputs under `artifacts/error_analysis/`:

- `validation_predictions.csv` — one row per validation image with score, prediction, TP/TN/FP/FN type and mask-area ratio;
- `score_distribution.png` — clean and defective score distributions;
- `defect_area_tp_vs_fn.png` — defect-size comparison between detected and missed positives;
- `false_negatives.png` — the most confidently missed defects, with the ground-truth mask contour;
- `false_positives.png` — the most confident false alarms.

This analysis is performed only on validation data. Its purpose is to diagnose the current baseline before choosing the next model change.
