# LoRAFlow

An independently implemented, modular pipeline for preparing datasets, fine-tuning language models with LoRA, evaluating base and adapted models, exporting adapters, and serving inference APIs.

> **Status:** Initial project scaffold. Training and serving integrations are being implemented and validated incrementally. This repository does not claim completed training results.

## Scope

LoRAFlow follows a practical fine-tuning lifecycle:

1. Validate and prepare instruction datasets.
2. Configure a base model and LoRA adapter.
3. Run supervised fine-tuning with Hugging Face tooling.
4. Evaluate the base and adapted models on the same held-out examples.
5. Save and optionally merge/export adapters.
6. Serve inference through a lightweight FastAPI application.
7. Track experiments and expose basic operational metrics.

## Deliberate exclusions

- Kubernetes and cluster orchestration
- vLLM serving

## Initial implementation priorities

- Reproducible configuration and data validation
- A clear separation between training, evaluation, export, and serving
- CPU/MPS-friendly development where supported
- CUDA-specific QLoRA dependencies kept optional rather than required on macOS
- Tests for project-owned logic before claiming end-to-end compatibility

## Repository layout

```text
LoRAFlow/
├── configs/
│   └── default.yaml
├── src/
│   └── loraflow/
│       ├── __init__.py
│       ├── config.py
│       └── data.py
├── tests/
├── pyproject.toml
└── README.md
```

## Development setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Install training and serving dependencies only when those components are needed:

```bash
pip install -e ".[train]"
pip install -e ".[serve]"
```

The optional `qlora` extra is intended for compatible Linux/CUDA environments. It is not part of the default macOS installation.

## Current status

The first commit establishes project metadata, typed configuration loading, and dataset validation. Model training, evaluation, adapter export, and API serving will be added as separately testable stages. No benchmark or accuracy number is claimed until it has been measured on a documented dataset and evaluation split.

## Reference

The high-level workflow was informed by the public project [Rishi625/LLM-Finetune-Pipeline](https://github.com/Rishi625/LLM-Finetune-Pipeline). LoRAFlow is a separate implementation; no source files are copied from that repository.
