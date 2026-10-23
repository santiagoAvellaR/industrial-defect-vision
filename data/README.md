# Data

M1 uses **Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)** as the primary dataset.

## Why this dataset

KSDD2 is a real industrial surface-inspection dataset with defect-free and defective samples plus pixel-level masks. That lets this project use the same data for classification first and segmentation later.

## Source and license

Official dataset page:

https://www.vicos.si/resources/kolektorsdd2/

KSDD2 is released under **CC BY-NC-SA 4.0**. The dataset is downloaded from the official source and is not committed to this repository.

## Local layout

The downloaded archive uses this structure:

```text
data/raw/ksdd2/
├── train/
│   ├── 10000.png
│   ├── 10000_GT.png
│   └── ...
└── test/
    ├── 20000.png
    ├── 20000_GT.png
    └── ...
```

Each sample image has a mask with the same stem plus `_GT`.

## Dataset inspection

Run:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

After removing two accidental local copies (`10301 (copy).png` and `10301_GT (copy).png`), the inspected dataset matches the official counts:

- train: 2,331 samples = 246 defective + 2,085 clean;
- test: 1,004 samples = 110 defective + 894 clean;
- total: 3,335 samples = 356 defective + 2,979 clean;
- no missing masks, orphan masks or image/mask size mismatches.

## Split policy

The official KSDD2 test set stays untouched for final evaluation.

The official training set is divided into train and validation subsets using a fixed seed and stratification by defect presence:

- train: 80% of the official training set;
- validation: 20% of the official training set;
- test: official KSDD2 test set.

Generate the manifests with:

```bash
python scripts/create_splits.py data/raw/ksdd2
```

Default settings:

- seed: `42`;
- validation fraction: `0.2`.

The command writes:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

Each row stores the relative image path, mask path and binary defect label.

## M1 checks

Before training any model we will verify:

- image and mask pairing;
- mask/image spatial alignment;
- positive/negative class balance;
- image sizes and channels;
- defect-area distribution;
- exact and obvious near-duplicates;
- reproducibility of the train/validation split;
- whether normalization or resizing is required.
