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
