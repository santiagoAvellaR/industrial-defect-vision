# Industrial Defect Vision

End-to-end Deep Learning project for industrial defect detection and semantic segmentation.

## Project goals

This repository is designed to combine:

- modern Computer Vision;
- Deep Learning with PyTorch;
- transfer learning;
- semantic segmentation;
- rigorous model evaluation and error analysis;
- experiment tracking;
- model inference;
- ONNX / ONNX Runtime;
- FastAPI serving;
- Docker;
- automated testing;
- CI.

The project is intentionally built as an ML engineering system rather than a notebook-only experiment.

## Status

**Current milestone: M1 — Dataset & exploration.**

M0 — repository foundation — is complete and validated by GitHub Actions on Python 3.11, 3.12 and 3.13.

The primary dataset selected for M1 is **KolektorSDD2**, an industrial surface-defect dataset with defective/defect-free samples and pixel-level annotations suitable for the project progression from classification to segmentation.

## Planned workflow

```text
Dataset
  ↓
Preprocessing & Augmentation
  ↓
PyTorch Training
  ↓
Evaluation & Error Analysis
  ↓
Experiment Tracking
  ↓
Best Model
  ↓
PyTorch / ONNX Inference
  ↓
FastAPI
  ↓
Docker
```

## Core milestones

- [x] M0 — Repository foundation
- [ ] M1 — Dataset & exploration
- [ ] M2 — PyTorch CNN baseline
- [ ] M3 — Transfer learning
- [ ] M4 — Semantic segmentation
- [ ] M5 — Modern pretrained segmentation
- [ ] M6 — MLflow experiment tracking
- [ ] M7 — Error analysis
- [ ] M8 — Inference & ONNX
- [ ] M9 — FastAPI service
- [ ] M10 — Docker & CI hardening
- [ ] M11 — Portfolio polish
- [ ] M12 — TensorFlow/Keras comparison (optional)

## Repository principles

- Prefer simple, explicit and readable implementations.
- Code should feel natural and idiomatic rather than artificially abstract or over-engineered.
- Add abstractions only when they solve a demonstrated problem.
- Reusable code belongs in `src/industrial_vision/`.
- Executable entry points belong in `scripts/` and should call reusable code from `src/`.
- Notebooks are for exploration, visualization and error analysis, not the only way to train or infer.
- Learning notes live in `docs/learning/` and should connect theory to real project evidence.
- Architecture decisions with meaningful trade-offs live in `docs/decisions/`.
- Human-readable experiment conclusions live in `docs/experiments/`.
- Large datasets, checkpoints and generated ML artifacts are not committed to Git.

## Local setup

Python 3.11 is the default development environment. The foundation CI also validates Python 3.12 and 3.13.

### Conda

```bash
conda env create -f environment.yml
conda activate industrial-defect-vision
pytest
```

`environment.yml` creates the Conda environment and installs the project from `pyproject.toml`, which remains the source of truth for Python dependencies.

### venv / pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Documentation

Project documentation lives under `docs/`:

- `docs/architecture/` — system architecture;
- `docs/decisions/` — Architecture Decision Records (ADRs);
- `docs/experiments/` — experiment summaries and conclusions;
- `docs/learning/` — durable technical learning notes;
- `docs/portfolio/` — project explanation and interview preparation.

Dataset source, license, split policy and preparation notes live in `data/README.md`.

## Reproducibility

Dataset and large model artifacts will not be committed directly to Git unless explicitly documented and license-compatible.

Instructions for obtaining data, reproducing experiments and running inference will be maintained as the project evolves.
