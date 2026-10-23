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

**Current milestone: M0 — Repository foundation (complete).**

The repository foundation has been implemented and validated by GitHub Actions on Python 3.11, 3.12 and 3.13. Dataset selection and all ML functionality start in M1 and later milestones.

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

- Reusable code belongs in `src/industrial_vision/`.
- Executable entry points belong in `scripts/` and should call reusable code from `src/`.
- Notebooks are for exploration, visualization and error analysis, not the only way to train or infer.
- Learning notes live in `docs/learning/` and should connect theory to real project evidence.
- Architecture decisions with meaningful trade-offs live in `docs/decisions/`.
- Human-readable experiment conclusions live in `docs/experiments/`.
- Large datasets, checkpoints and generated ML artifacts are not committed to Git.

## Local setup

Python 3.11–3.13 is supported for the repository foundation.

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

## Reproducibility

Dataset and large model artifacts will not be committed directly to Git unless explicitly documented and license-compatible.

Instructions for obtaining data, reproducing experiments and running inference will be maintained as the project evolves.
