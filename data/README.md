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

Each sample image should have a mask with the same stem plus `_GT`.

## Initial inspection

The current local archive contains:

- 6,672 PNG files;
- 3,336 RGB images and 3,336 grayscale images;
- 4,664 files under `train`;
- 2,008 files under `test`;
- 356 non-empty mask candidates;
- no unreadable image files.

This suggests 2,332 apparent training samples and 1,004 test samples. The official dataset summary reports 2,331 training samples and 1,004 test samples, so the local training set contains one extra or irregular sample that must be identified before the split is frozen.

Run the project inspector with:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

## Split policy

The official test set stays untouched for final evaluation.

After the extra training sample is understood, the official training set will be divided into train and validation subsets with a fixed seed and stratification by defect presence.

Target proportions inside the official training set:

- train: 80%;
- validation: 20%;
- test: official KSDD2 test set.

The resulting manifests will be stored under `data/splits/`.

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
