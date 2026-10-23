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

## Dataset validation

The validated dataset contains:

- 3,335 samples;
- 356 defective samples;
- 2,979 defect-free samples;
- official train set: 246 defective + 2,085 clean;
- official test set: 110 defective + 894 clean;
- no image/mask size mismatches.

During inspection, two accidental local copies were found in `train/`:

- `10301 (copy).png`;
- `10301_GT (copy).png`.

They were removed before creating the manifests because they are not part of the official dataset.

Run the validation with:

```bash
python scripts/inspect_dataset.py data/raw/ksdd2
```

## Split policy

The official test set remains untouched for final evaluation.

The official training set is split deterministically with seed `42`, stratified by defect presence:

- train: 1,865 samples = 197 defective + 1,668 clean;
- validation: 466 samples = 49 defective + 417 clean;
- test: 1,004 samples = 110 defective + 894 clean.

Generate the manifests with:

```bash
python scripts/create_splits.py data/raw/ksdd2
```

The generated files are stored in `data/splits/`:

```text
train.csv
val.csv
test.csv
```

Each row contains:

```text
image,mask,label
```

where `label=1` means the mask contains a defect and `label=0` means the mask is empty.

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
