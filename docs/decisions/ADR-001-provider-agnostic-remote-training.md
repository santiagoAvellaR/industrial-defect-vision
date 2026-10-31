# ADR-001 — Provider-agnostic remote GPU training

Status: Accepted

## Context

M4 segmentation training became slow on the local CPU. A cloud GPU is useful, but coupling model code to one provider would make the project harder to reproduce and migrate.

A Camber Student environment with an NVIDIA L4 was validated successfully. The same PyTorch training script ran on CUDA without framework-specific changes. The main performance issue was not CUDA but reading thousands of PNG files directly from Camber Stash.

## Decision

Keep training code provider-agnostic.

The training script selects the available PyTorch device:

```text
CUDA available -> GPU
otherwise      -> CPU
```

Cloud providers are treated as infrastructure only. Source code moves through Git/GitHub. Large datasets and model artifacts remain outside Git.

For Camber:

```text
persistent Stash
      |
      | stage once per compute session
      v
local node storage (/tmp/ksdd2)
      |
      v
PyTorch DataLoader -> NVIDIA L4
```

Checkpoints and other outputs that must survive the compute session are written back to persistent storage.

DataLoader performance settings such as `num_workers` and pinned memory are configurable runtime parameters rather than Camber-specific code.

## Alternatives considered

### Train locally only

Rejected for expensive segmentation experiments because CPU runtime is too high and slows the experiment loop.

### Provider-specific training implementation

Rejected because Camber, Lightning, Kaggle, UTC HPC or another GPU backend should be replaceable without changing the ML architecture or training loop.

### Read KSDD2 directly from remote Stash during training

Rejected for the current PNG-based dataset because the remote filesystem caused severe small-file I/O overhead and left the GPU underfed.

## Consequences

### Positive

- The same training command works locally and remotely.
- Compute providers remain replaceable.
- Remote execution resembles a professional ML workflow: Git for code, external storage for data/artifacts, GPU compute for training.
- DataLoader performance can be benchmarked separately from model experiments.

### Negative

- A compute session requires environment verification and dataset staging before training.
- `/tmp` is ephemeral, so important artifacts must not remain there.
- Infrastructure benchmarks add a small operational step before large experiments.

## Evidence / references

A full SimpleUNet epoch plus validation on an NVIDIA L4 took about 35 seconds when KSDD2 was staged to `/tmp`. Reading the same thousands of PNG files directly from Camber Stash was dramatically slower, confirming remote small-file I/O as the bottleneck rather than CUDA.
