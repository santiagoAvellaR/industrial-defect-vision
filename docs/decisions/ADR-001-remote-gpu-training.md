# ADR-001 — Use remote GPU compute for training

Status: Accepted

## Context

The M4 SimpleUNet segmentation experiments are currently executed on local CPU. As training duration increases, this makes controlled experimentation unnecessarily slow and ties experiments to one laptop.

The training code already selects CUDA automatically when available, so remote GPU execution does not require a different modeling implementation. The repository also already keeps raw datasets and large model artifacts out of Git.

## Decision

Use remote GPU compute for expensive training runs while keeping the same repository, scripts, experiment definitions and checkpoints semantics.

Initial execution backend: **Lightning AI Studio**.

Workflow:

```text
local laptop
    |
    | Git / SSH
    v
remote Studio
    |
    +-- repository code
    +-- reproducible Python environment
    +-- persistent KSDD2 data
    +-- GPU training
    +-- checkpoints / figures / logs
```

Source code remains synchronized through Git. Raw data and large generated artifacts remain outside Git and live in remote persistent storage or are copied back explicitly.

The remote backend is an execution detail, not part of model behavior. The same training command must continue to work locally on CPU and remotely on CUDA.

## Alternatives considered

### Kaggle Notebooks

Useful free GPU fallback with very low setup cost. Not selected as the primary workflow because the notebook-centric execution model is less similar to a normal repository + remote-machine engineering workflow.

### Camber Student

Useful additional student GPU quota through the GitHub Student Developer Pack. Kept as a secondary option.

### UTC PILCAM2 / BMBI compute

Potentially the strongest academic option. UTC operates PILCAM2 and BMBI has NVIDIA GH200 systems. Access for this project is not assumed and should be requested through UTC. If access is granted, the repository should later support the same workflow through SSH / scheduler-based execution.

### Local CPU

Remains useful for tests, debugging, threshold sweeps and small checks, but not as the default for long training runs.

## Consequences

### Positive

- much faster experiment iteration;
- experience with remote ML development closer to professional workflows;
- local hardware no longer limits model experimentation;
- code remains portable because training scripts are not cloud-specific;
- prepares the project for later MLflow, artifact storage and HPC workflows.

### Negative

- remote data and artifacts need explicit lifecycle management;
- free GPU quotas and availability can change;
- remote environments introduce another reproducibility surface to verify;
- cloud credits must not be wasted on installation, data upload or idle sessions.

## Evidence / references

- `scripts/train_simple_unet.py` automatically uses CUDA when available.
- `.gitignore` excludes raw datasets and large model artifacts from Git.
- EXP-013 required a long CPU run and demonstrated that longer training can improve validation performance, making faster execution operationally useful.
