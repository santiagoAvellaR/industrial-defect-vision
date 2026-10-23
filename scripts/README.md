# Scripts

Executable entry points should remain thin and call reusable code from `src/industrial_vision/`.

## Dataset inspection

After extracting the dataset locally, inspect its real structure before implementing dataset-specific loading logic:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

The reusable inspection logic lives in `src/industrial_vision/data/inspection.py`.
