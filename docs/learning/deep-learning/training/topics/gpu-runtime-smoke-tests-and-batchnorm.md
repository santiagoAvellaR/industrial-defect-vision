# Concept — GPU runtime, CUDA OOM, smoke tests and BatchNorm batch constraints

## What is it?

This note groups several runtime concepts that became concrete while moving M5 training to Google Colab:

- what a GPU such as NVIDIA T4 is;
- what CUDA means in PyTorch;
- what `CUDA out of memory` means;
- why we run smoke tests before expensive experiments;
- why BatchNorm can fail on a singleton training batch;
- why `drop_last=True` was the minimal fix for EXP-014.

## Why do we need it in this project?

Modern segmentation models are much heavier than `SimpleUNet`. A correct experiment is not only about model code: the runtime must fit the model in GPU memory, DataLoader behavior must be compatible with the architecture, and failures must be detected before launching a 20-epoch run.

This became visible when the first DeepLabV3 smoke test failed even though the GPU had enough memory.

## Mental model

```text
Colab runtime
   ↓
NVIDIA GPU (T4 in our current session)
   ↓
CUDA
   ↓
PyTorch tensors + kernels
   ↓
model forward/backward
```

The GPU is the physical accelerator. CUDA is NVIDIA's compute platform that lets frameworks such as PyTorch execute operations on that GPU.

## Technical details

### GPU model names

Names such as T4, L4, A100 and H100 refer to different NVIDIA GPU products and generations.

A useful mental scale is:

- **T4** — older datacenter GPU, still useful for moderate training and inference;
- **L4** — newer and generally faster/more efficient for modern AI workloads;
- **A100** — high-end datacenter training GPU;
- **H100** — newer high-end accelerator aimed at very large training workloads.

The exact best GPU depends on model size, memory needs, cost and availability. More powerful hardware does not change the scientific question of the experiment; it mainly changes how fast and how large a workload can run.

In our Colab session, PyTorch reported a Tesla T4 with about 14.56 GiB available to the runtime.

### CUDA OOM

OOM means **Out Of Memory**.

A CUDA OOM occurs when the GPU cannot allocate enough VRAM for an operation. During training, VRAM is needed for more than just model weights:

```text
VRAM
├─ model parameters
├─ activations saved for backward
├─ gradients
├─ optimizer state
└─ current batch tensors
```

Typical ways to reduce VRAM pressure include:

- reduce batch size;
- reduce image resolution;
- use mixed precision;
- use gradient accumulation;
- choose a smaller model;
- freeze parts of the model when scientifically appropriate.

A CUDA OOM is different from normal system RAM exhaustion.

### Smoke test

A smoke test is a cheap end-to-end check run before a more expensive experiment.

The name comes from hardware engineering: power something on and first check whether it immediately fails badly — metaphorically, whether it produces smoke.

For this project, a one-epoch GPU smoke test verifies that the complete path works:

```text
dataset
  ↓
DataLoader
  ↓
GPU transfer
  ↓
forward
  ↓
loss
  ↓
backward
  ↓
optimizer step
  ↓
validation
  ↓
checkpoint save
```

It does **not** prove the model is good. It proves the system is healthy enough to justify a longer run.

### The EXP-014 BatchNorm failure

The first DeepLabV3 smoke test did not fail because of CUDA OOM.

The final training batch contained only one sample. Inside DeepLabV3's ASPP branch, global pooling produced an activation shaped roughly like:

`[1, 256, 1, 1]`.

BatchNorm in training mode needs enough values per channel to estimate batch statistics. With batch size `1` and spatial size `1 x 1`, each channel had only one value, so PyTorch raised:

`Expected more than 1 value per channel when training`.

### Why `drop_last=True` fixes it

For the DeepLabV3 training loader we use:

```python
DataLoader(..., drop_last=True)
```

If the dataset size is not divisible by the batch size, the incomplete final training batch is discarded.

In EXP-014 this drops one training sample per epoch and prevents a singleton batch from reaching BatchNorm.

Validation still uses `drop_last=False`, because BatchNorm is in evaluation mode there and we want metrics over the complete validation set.

## Where is it used in the repository?

- Colab runtime workflow: `notebooks/colab/remote_training.ipynb`
- DeepLab training loader: `scripts/train_deeplabv3.py`
- Shared training loop: `src/industrial_vision/training/segmentation.py`
- Experiment record: `docs/experiments/EXP-014-deeplabv3-resnet50-pretrained.md`

## Alternatives

For the singleton BatchNorm issue, alternatives include:

- freeze BatchNorm layers;
- replace BatchNorm with GroupNorm or another normalization;
- choose a batch size that divides the training set exactly;
- change the architecture.

Those options modify more of the model behavior. `drop_last=True` was the smallest useful fix for this experiment.

## Trade-offs

`drop_last=True`:

- keeps BatchNorm training valid;
- preserves the intended batch size;
- changes almost nothing architecturally;
- but omits a small number of training samples each epoch.

Reducing batch size can reduce VRAM use, but it can also change optimization dynamics and make BatchNorm less stable.

## Common mistakes

- calling every GPU-related failure a CUDA OOM;
- confusing GPU VRAM with CPU/system RAM;
- assuming a successful forward pass guarantees backward will fit in memory;
- launching a long experiment without a smoke test;
- using `drop_last=True` on validation/test loaders and silently excluding samples from metrics;
- ignoring the effect of BatchNorm when batch sizes become very small.

## Interview questions

1. What is the difference between a GPU, CUDA and PyTorch?
2. What consumes GPU memory during training besides model parameters?
3. Why can BatchNorm fail with batch size one?
4. Why is `drop_last=True` acceptable for training but usually not for validation?
5. What does a smoke test prove, and what does it not prove?
6. How would you debug a CUDA OOM systematically?

## Evidence

- First DeepLab smoke test: failed on a singleton BatchNorm input `[1, 256, 1, 1]`.
- Fix: training DataLoader uses `drop_last=True`.
- Second smoke test: completed one full train/validation epoch and saved best-loss and best-Dice checkpoints.
- Runtime: Google Colab, NVIDIA Tesla T4.
