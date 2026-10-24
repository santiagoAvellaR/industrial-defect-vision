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
