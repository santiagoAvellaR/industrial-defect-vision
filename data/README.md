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

## Exploratory findings

The training images are similar in shape but not identical:

- 479 unique image sizes;
- width range: 184–241 pixels;
- height range: 602–658 pixels;
- common sizes are close to 230 × 638 pixels.

Defects are usually small relative to the full image. Among the 197 positive training samples:

- median defect area: 1.535% of pixels;
- minimum defect area: 0.017%;
- maximum defect area: 20.583%.

This means pixel accuracy alone would be misleading for segmentation because a model could predict mostly background and still obtain a high value.

## Initial preprocessing decision

The first PyTorch pipeline uses a fixed size of **640 × 224 pixels (height × width)**.

Why:

- it is close to the typical original geometry;
- both dimensions are divisible by 32, which is convenient for CNN encoder/decoder architectures;
- fixed spatial dimensions allow samples to be stacked into batches;
- cropping is avoided because very small defects could be removed entirely.

Images are resized with bilinear interpolation. Masks are resized with nearest-neighbor interpolation so class labels remain discrete. Images are converted to float tensors in `[0, 1]`; masks are converted to binary tensors.

This is a simple baseline preprocessing choice and can be revisited if later experiments show that aspect-ratio distortion is harmful.

## M1 checks

Completed:

- image and mask pairing;
- mask/image spatial alignment;
- positive/negative class balance;
- image-size exploration;
- defect-area distribution;
- reproducible train/validation split;
- initial resize strategy;
- PyTorch image/mask loader.

Still to check before closing M1:

- exact and obvious near-duplicates;
- a few real samples through the PyTorch loader and DataLoader;
- final sample visualizations after preprocessing.
