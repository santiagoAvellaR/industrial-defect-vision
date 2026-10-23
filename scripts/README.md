# Scripts

Executable entry points should remain thin and call reusable code from `src/industrial_vision/`.

## KSDD2 inspection

After extracting KSDD2 locally, run:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

The script checks image/mask pairing, defective versus clean samples, and image/mask size mismatches for the official `train` and `test` folders.
