# Remote GPU training workflow

## Goal

Run the existing PyTorch training pipeline on remote GPU compute without introducing provider-specific ML code.

Current validated provider: Camber Cloud Student with one NVIDIA L4.

## Separation of responsibilities

```text
Git/GitHub
-> source code, tests and documentation

Persistent storage
-> raw dataset archives, checkpoints and durable artifacts

Ephemeral node storage
-> staged dataset used during active training

Remote GPU
-> model training and validation
```

The repository intentionally ignores raw datasets and model weights.

## Camber session startup checklist

From the remote project directory:

```bash
git pull --ff-only
python -m pip install -e ".[dev]"
python -m pytest
python -c "import industrial_vision"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Do not assume that switching the Camber compute node automatically switches an already running Jupyter server. If `nvidia-smi` is unavailable or `torch.cuda.is_available()` is false, restart the Jupyter server from JupyterHub on the GPU compute before training.

The editable Python installation may also need to be repeated after a server/compute restart even when project files persist.

## Dataset staging

Reading KSDD2 directly from Camber Stash caused severe small-file I/O overhead because the dataset contains thousands of PNG files. The validated approach is to stage KSDD2 to local node storage before training.

### Direct-copy method

If the persistent dataset exists as `data/raw/ksdd2`:

```bash
rm -rf /tmp/ksdd2
cp -a data/raw/ksdd2 /tmp/ksdd2
```

Train against:

```text
/tmp/ksdd2
```

A full SimpleUNet epoch plus validation took about 35 seconds on the NVIDIA L4 after staging to `/tmp`, while direct Stash access was dramatically slower.

### Recommended archive method

Create the archive once after a correct KSDD2 copy is available:

```bash
tar -cf data/raw/ksdd2.tar -C /tmp ksdd2
```

On future sessions, transfer one large sequential file and extract it locally:

```bash
rm -rf /tmp/ksdd2 /tmp/ksdd2.tar
cp data/raw/ksdd2.tar /tmp/ksdd2.tar
tar -xf /tmp/ksdd2.tar -C /tmp
```

Verify:

```bash
find /tmp/ksdd2 -type f | wc -l
```

The archive is infrastructure data and must remain outside Git.

## DataLoader runtime controls

`train_simple_unet.py` exposes:

```text
--batch-size
--num-workers
--pin-memory
```

Defaults remain conservative and preserve previous local behavior:

```text
batch_size=4
num_workers=0
pin_memory=False
```

When `--pin-memory` is enabled on CUDA, host tensors are pinned and training/evaluation use non-blocking device transfers.

`num_workers` and pinned-memory settings are infrastructure/runtime controls. `batch_size` is different: it also changes optimization dynamics because it changes how many samples contribute to each optimizer update.

## Infrastructure benchmark

Before the next expensive experiment, benchmark short disposable runs using `/tmp/ksdd2` while keeping `batch_size=4` fixed.

Recommended sequence:

```text
A. num_workers=0, pin_memory=off   <- baseline
B. num_workers=2, pin_memory=off
C. num_workers=4, pin_memory=off
D. best worker count, pin_memory=on
```

This isolates DataLoader/transfer performance without changing the mathematical batch size used in EXP-013.

Example:

```bash
time python scripts/train_simple_unet.py /tmp/ksdd2 \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 2 \
  --pin-memory \
  --learning-rate 3e-4 \
  --threshold 0.5 \
  --dice-positive-only \
  --checkpoint /tmp/benchmark_best_loss.pt \
  --dice-checkpoint /tmp/benchmark_best_dice.pt
```

Record at least:

- wall-clock time;
- GPU utilization from `nvidia-smi` while the run is active;
- observed/peak VRAM;
- worker count;
- pin-memory setting.

The resulting model metrics are not an experiment result; these runs exist only to measure infrastructure performance.

### Batch-size benchmarking

Only consider `batch_size=8` or `16` after the DataLoader benchmark if GPU utilization or VRAM measurements suggest substantial unused capacity.

Because batch size can change optimization behavior, adopting a different batch size for future model training should be treated as an explicit experimental choice rather than a transparent infrastructure optimization.

## Running a real experiment

Use the same command locally or remotely; only the dataset path and runtime-performance flags may differ.

Example on Camber:

```bash
python scripts/train_simple_unet.py /tmp/ksdd2 \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 2 \
  --pin-memory \
  --learning-rate 3e-4 \
  --threshold 0.5 \
  --dice-positive-only \
  --checkpoint artifacts/checkpoints/example_best_loss.pt
```

The modeling code remains standard PyTorch. Camber is infrastructure, not a project dependency.

## Durable outputs

Never rely on `/tmp` for artifacts that must survive the node/session.

After a real experiment, keep checkpoints and figures in persistent project storage, for example:

```text
artifacts/checkpoints/
artifacts/figures/
```

Then stop the GPU compute when training and analysis are complete.

## Provider portability

The same structure should work with another remote backend later:

```text
Git -> remote environment -> local-to-node dataset staging -> PyTorch CUDA -> persistent artifacts
```

A future UTC/HPC integration may add scheduler-specific run commands, but it should not modify model architecture or training logic.
