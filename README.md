# LoRAFlow

[![CI](https://github.com/Sujeet-Kona/LoRAFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujeet-Kona/LoRAFlow/actions/workflows/ci.yml)

**A configuration-driven LLM fine-tuning and evaluation pipeline** for dataset validation, supervised fine-tuning with standard LoRA, held-out evaluation, adapter export, and API-based inference.

> **Scope:** LoRAFlow demonstrates an end-to-end fine-tuning workflow on controlled synthetic tasks and a 0.5B model. It is a single-process, single-device engineering project—not evidence of broad language-model generalization or production-scale serving. QLoRA is not implemented.

## Pipeline overview

~~~mermaid
flowchart LR
    CONFIG[Config + CLI] --> DATA[Dataset validation]
    DATA --> TRAIN[Supervised fine-tuning]
    TRAIN --> ADAPTER[PEFT LoRA adapter]
    ADAPTER --> EVAL[Held-out evaluation]
    EVAL --> REPORT[Metrics + report]
    ADAPTER --> EXPORT[Export + reload verification]
    EXPORT --> API[FastAPI inference]
    TRAIN --> TRACK[MLflow tracking]
    EVAL --> TRACK
    EXPORT --> TRACK
    CI[GitHub Actions] --> CHECKS[Lint + tests + CPU smoke]
~~~

Training, evaluation, and export are offline workflows. The serving process loads an existing base model and adapter; it does not train during inference. MLflow records experiment parameters, metrics, and artifacts outside the request path.

## Current controlled benchmark: Prompt-v2

Prompt-v2 contains three controlled synthetic tracks. Each track has 600 training, 120 validation, and 150 held-out test examples. The primary baseline is train-only TF-IDF + LinearSVC. The unadapted Qwen model often emits prose that violates the required one-line output schema; its zero exact-match result is format-confounded and is not evidence that the adapter beats a capable base-model classifier.

| Track | LoRA exact match | TF-IDF + LinearSVC exact match | Structured-output validity | Category macro-F1 |
|---|---:|---:|---:|---:|
| A | 68.00% | 30.00% | 100.00% | 0.7004 |
| B | **77.33%** | 56.00% | 98.67% | **0.8606** |
| C | 37.33% | 16.67% | 100.00% | 0.4345 |

Track B is the strongest result in this benchmark. Track C shows a substantial category-generalization weakness and should not be presented as a strong classifier.

### Leakage and split audit

- Exact duplicate notes in each test split: zero.
- Track A: 21/150 test notes (14.00%) had a training neighbour with word-level Jaccard similarity ≥ 0.9; mean best similarity was 0.8239. This overlap weakens a clean unseen-concept generalization claim.
- Tracks B and C: zero test notes at or above the 0.9 Jaccard threshold.
- The audit compares every test note against all 600 training notes in its track.

These are synthetic-task measurements, not evidence of veterinary utility, clinical safety, or broad natural-language generalization. The action field is deterministically derived from severity and is not an independent prediction task. Full field metrics, per-class results, split hashes, MLflow run IDs, and limitations are in [Prompt-v2 results](docs/prompt-v2-results.md). The multi-seed runner and its scope are described in [Multi-seed evaluation](docs/multi-seed-evaluation.md); no multi-seed robustness result is claimed here.

### Reproduce Prompt-v2

The JSONL inputs, manifests, and configurations are committed. A fresh run requires the train, serve, and dev extras plus the Hugging Face model download.

~~~bash
pip install -e ".[train,serve,dev]"
bash scripts/run_prompt_v2_experiments.sh
~~~

The runner verifies split hashes and counts, trains and evaluates the three tracks, exports and reload-validates adapters, and runs local serving smoke checks. It refuses to overwrite existing run/export directories.

## Historical v1 results — retained with leakage disclosure

The original v1 synthetic corpus used label-correlated phrase templates. The dataset audit found 65–71% template overlap in held-out splits and an 82.5–85% 1-nearest-neighbour exact-match baseline. Therefore, the v1 model metrics below are retained only as historical pipeline and reproducibility evidence; they are **not valid evidence of task generalization**.

| Measurement | Historical v1 result |
|---|---|
| Base model / adapter | Qwen2.5-0.5B; standard LoRA rank 8, alpha 16, q_proj/v_proj |
| Trainable parameters | 540,672 / 494,573,440 (0.109%) |
| Training | 3 epochs, 90 steps, MPS, fp32, seed 42; 413 s |
| Training / best evaluation loss | 0.8618 / 0.1346336007118225 |
| Held-out answer-token loss | 4.0200 → 0.1490 |
| Perplexity | 55.6996 → 1.1606 |
| Answer-token accuracy | 0.4645 → 0.9551 |
| Exact match | 0.0000 → 0.5000 |
| 3-token prefix match | 0.0000 → 0.5000 |
| Per-field generation accuracy (24 greedy outputs) | severity 0.0 → 0.875; category 0.0 → 0.5417; action 0.0 → 0.875 |
| Merged export | 494,032,768 parameters; max logit delta 0.000303 against tolerance 0.05 |
| Local serving | /health, /generate, and /metrics verified; malformed input 422, oversized prompt 400, unknown route 404 |
| Experiment tracking | Five finished MLflow runs: two training and three evaluation |
| Reproducibility | Repeated 90-step run matched loss, checkpoint, adapter hash, and evaluation report; wall clock differed |

The v1 run was repeated into a separate output directory. The two runs matched best_eval_loss (0.1346336007118225), train_loss (0.8617594838142395), selected checkpoint (checkpoint-90), adapter weights, and evaluation report. Adapter and merged exports were also repeated; the merged export retained a maximum logit delta of 0.000303 against tolerance 0.05. These results verify repeatability of that historical pipeline run, not generalization.

## Dataset development and audit history

- **v1:** label-correlated templates; held-out overlap and a strong nearest-neighbour baseline invalidated generalization claims.
- **v2:** audit exposed a closed-vocabulary shortcut because symptom concepts appeared across splits.
- **v3:** generator introduced category-specific symptom-concept holdouts and derived severity from independently sampled systemic, functional, and trajectory attributes. The committed local audit reported 480/120/120 records, disjoint symptom concepts, no exact-note or template overlap, and passed category/severity/action consistency checks. No v3 model-quality result is claimed here.
- **Prompt-v2:** the three-track held-out measurements are reported above, with the Track A lexical-overlap caveat.

The synthetic animal-observation task is an engineering fixture for evaluating structured-output behavior and controlled distribution shifts. It is not veterinary guidance or a clinical decision system.

## Implementation

### Fine-tuning and evaluation

- Configuration validation and JSONL dataset checks.
- Supervised fine-tuning with Hugging Face Transformers, TRL, and PEFT LoRA.
- Held-out evaluation using the same examples for base and adapted model comparisons.
- Exact match, prefix match, per-field metrics, token-level metrics, and structured-output validation.
- Reproducible run metadata, split manifests, and experiment tracking through MLflow.
- Adapter export with reload checks; merged export with adapter-versus-merged logit comparison.

The verified training path uses standard LoRA. QLoRA, FSDP, DeepSpeed, multi-node training, and multi-process training are not implemented.

### Inference API

The FastAPI service exposes /health, /generate, and /metrics. Prompt-v2 serving smoke tests passed locally for all three exported adapters: each health check returned ok, the service reported adapter_applied: true, and generated a schema-valid completion. This verifies the tested local path, not production deployment.

The API validates request bodies, rejects over-length prompts, reports model-load failures rather than silently falling back, and exports request/generation metrics. The service is single-host and does not implement authentication or rate limiting.

### CI and container status

GitHub Actions runs lint, pure-logic tests on Python 3.10 and 3.13, and a CPU pipeline smoke using sshleifer/tiny-gpt2: dataset build, validation, two-step training, evaluation, export, and live API checks. The tiny-model job validates pipeline wiring; its metrics are not a model-quality result.

The Dockerfile and Compose configuration are present, but the image build and live container stack have not been verified because Docker was unavailable in the recorded development environment. Do not interpret CI configuration validation as a successful container deployment.

## Quickstart

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core logic and tests
pip install -e ".[train,serve]"  # model training and inference
~~~

Typical CLI workflow:

~~~bash
loraflow data-check --config configs/train.yaml
loraflow train      --config configs/train.yaml
loraflow evaluate   --config configs/train.yaml
loraflow export     --config configs/train.yaml
loraflow serve      --config configs/train.yaml --port 8125
pytest
~~~

For the controlled Prompt-v2 benchmark, use the dedicated reproduction command above. Existing export directories are protected from accidental overwrite; use the explicit overwrite option only when replacement is intentional.

## Technology

Python · PyTorch · Hugging Face Transformers · TRL · PEFT (LoRA) · FastAPI · MLflow · pytest · Ruff · GitHub Actions.

## Repository guide

- src/loraflow/ — configuration, runtime, data, metrics, training, evaluation, tracking, export, API, and CLI.
- configs/ — training, smoke, CI, and reproducibility configurations.
- scripts/ — dataset generation, audits, experiment runners, and serving smoke tests.
- tests/ — unit and integration coverage.
- docs/prompt-v2-results.md — benchmark protocol and detailed results.
- docs/multi-seed-evaluation.md — multi-seed runner and methodology.
- docs/architecture.md — architecture and pipeline boundaries.

## Limitations

- Reported model-quality results are from small synthetic tasks and a 0.5B model; no public benchmark is reproduced.
- There is no human evaluation, semantic-equivalence assessment, or calibration metric.
- Training is single-process and single-device.
- Serving is a correctness-checked single-host service, not throughput-tuned; it has no authentication or rate limiting.
- MLflow uses a local SQLite store.
- Lower-precision merged exports use a wider comparison tolerance; equivalence is not established to the same standard as fp32/bf16.
- Docker image build and live Compose execution remain unverified.
- QLoRA is not implemented.

## Reference

The high-level workflow was informed by [Rishi625/LLM-Finetune-Pipeline](https://github.com/Rishi625/LLM-Finetune-Pipeline). LoRAFlow is a separate implementation; no source files are copied from that repository.
