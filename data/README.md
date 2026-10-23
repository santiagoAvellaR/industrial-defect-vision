# Data

M1 uses **Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)** as the primary dataset.

## Why this dataset

KolektorSDD2 is a real industrial surface-inspection dataset captured in a controlled production environment. It contains both defective and defect-free samples and supports the progression used in this project: image-level defect classification first, then pixel-level defect localization/segmentation.

Dataset summary from the official ViCoS source:

- 3,335 images total;
- 356 images with visible defects;
- 2,979 images without defects;
- image size approximately 230 x 630 pixels;
- official training set: 246 positive + 2,085 negative images;
- official test set: 110 positive + 894 negative images;
- defects include scratches, minor spots and other surface imperfections.

## Source

Official dataset page:

https://www.vicos.si/resources/kolektorsdd2/

Download the dataset from the official source rather than redistributing it through this repository.

## License

KolektorSDD2 is released under **CC BY-NC-SA 4.0**. Commercial use requires permission from the dataset authors. The dataset itself is therefore not committed to this repository.

## Planned local layout

```text
data/
├── raw/          # original downloaded/extracted dataset; ignored by Git
├── processed/    # generated representations if needed; ignored by Git
└── splits/       # reproducible split manifests tracked by Git
```

The exact layout inside `raw/` will be documented after inspecting the downloaded archive instead of assuming a structure in advance.

## Split policy

The official test set will remain untouched for final evaluation.

The official training set will be divided into train and validation subsets using a deterministic, fixed-seed split stratified by defect presence. Before freezing that split, M1 will inspect filenames, duplicates and any grouping information that could create leakage between related samples.

Planned proportions inside the official training set:

- train: 80%;
- validation: 20%;
- test: official KSDD2 test set.

The resulting manifests will be stored under `data/splits/`.

## M1 checks

Before training any model we will verify:

- image and mask pairing;
- mask/image spatial alignment;
- image sizes and channels;
- positive/negative class balance;
- defect-area distribution;
- exact and obvious near-duplicates;
- reproducibility of the train/validation split;
- whether normalization or resizing is required.
