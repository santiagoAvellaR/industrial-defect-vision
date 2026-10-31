# Remote training workflow

## Goal

Run expensive PyTorch training on remote GPU compute without changing the modeling code or turning the project into a notebook-only workflow.

The repository stays the source of truth for code and experiment configuration. Raw datasets and large artifacts stay outside Git.

## Initial backend: Lightning AI Studio

Lightning is the first remote backend because it supports persistent environments, persistent/shared storage, GPU switching and SSH/local-IDE access while preserving a normal repository workflow.

Current provider quotas and pricing are external operational details and should be checked before each period of use.

## One-time setup

### 1. Create a Studio on CPU

Do setup work on the free CPU machine. Only switch to GPU when training is ready.

### 2. Clone the repository

```bash
git clone https://github.com/santiagoAvellaR/industrial-defect-vision.git
cd industrial-defect-vision
git switch feat/m4-segmentation-threshold-sweep
```

After the feature branch is merged, use `main` instead.

### 3. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the environment:

```bash
python -m pytest
```

### 4. Upload KSDD2 once

Keep KSDD2 in persistent remote storage, not in Git.

Expected project path:

```text
data/raw/ksdd2/
```

If the provider stores uploads elsewhere, create a symlink or copy into the expected path so training commands remain provider-independent.

Example:

```bash
mkdir -p data/raw
ln -s /teamspace/uploads/ksdd2 data/raw/ksdd2
```

Adjust the source path to the actual uploaded dataset location.

## Before every GPU run

Pull the exact code to execute:

```bash
git status
git pull --ff-only
```

Run focused tests on CPU before consuming GPU credits:

```bash
python -m pytest tests/test_training_segmentation.py tests/test_model_loading_simple_unet.py
```

Then switch the Studio to a GPU.

Verify CUDA:

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The training script should print:

```text
device: cuda
```

If it prints `device: cpu`, stop and fix the environment before launching a long run.

## Running an experiment

Use the same command that would be used locally. Example:

```bash
python scripts/train_simple_unet.py data/raw/ksdd2 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --threshold 0.5 \
  --dice-positive-only \
  --checkpoint artifacts/checkpoints/example_best_loss.pt
```

Do not change batch size only because a larger GPU is available unless batch size itself is the controlled experimental variable. Reproducible comparisons require keeping experiment variables explicit.

## Artifacts

Git tracks:

- source code;
- manifests and small configuration files;
- experiment documentation;
- architecture decisions.

Git does not track:

- raw KSDD2 data;
- `.pt`, `.pth`, `.ckpt` weights;
- other large generated model artifacts.

After a useful run, preserve the best checkpoints and figures in persistent remote storage and copy important artifacts back when needed. MLflow/artifact storage will later formalize this process.

## Development from the laptop

Preferred professional-style workflow:

```text
local VS Code / terminal
        |
        | SSH
        v
remote Lightning Studio
        |
        +-- code executes remotely
        +-- data lives remotely
        +-- GPU lives remotely
```

This keeps the familiar editor while moving compute away from the laptop.

## Fallbacks

- **Kaggle**: quick free GPU fallback for isolated runs.
- **Camber Student**: additional student GPU quota through GitHub Education.
- **UTC PILCAM2 / BMBI**: academic/HPC route to investigate in parallel. If access is granted, add a separate scheduler runbook rather than modifying model code for UTC.

## Reproducibility rule

Changing where an experiment runs should not change its scientific definition.

For a controlled comparison, preserve:

- dataset split;
- seed;
- architecture;
- loss;
- learning rate;
- batch size;
- epoch count;
- threshold used during reporting.

Record hardware/device separately as experiment metadata.
