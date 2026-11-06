# LoRAFlow

An independently implemented, modular pipeline for preparing datasets, fine-tuning language models with LoRA, evaluating base and adapted models, exporting adapters, and serving inference APIs.

> **Status:** The pipeline runs end to end and has been verified on an Apple M4 (16 GB) with MPS and CPU. Training, evaluation, tracking, export, and serving each completed real runs whose numbers are recorded below and on disk. It is **not** production-ready: the task is a small synthetic corpus, the model is a 0.5B toy, and single-host only.

## Measured results

Every figure below was printed by a real run on this machine and is on disk:
`artifacts/{train-run,repro-run}/training_metadata.json`, `artifacts/{train-run,repro-run}/evaluation_report.json`,
and the `export_manifest.json` files under the export directories.
No external benchmark is reproduced and no accuracy is claimed beyond this synthetic task.

| Stage | Measured on this machine |
| --- | --- |
| LoRA training | `Qwen/Qwen2.5-0.5B`, rank 8 / alpha 16 on `q_proj,v_proj`; 540,672 trainable of 494,573,440 params (0.109%); 3 epochs = 90 steps; `mps` + `torch.float32`; seed 42; 413 s wall clock; `train_loss` 0.8618; best held-out `eval_loss` 0.1346336007118225 at `checkpoint-90` |
| Baseline vs fine-tuned (same 120 held-out examples) | mean answer-token loss `4.0200 → 0.1490`; perplexity `55.6996 → 1.1606`; answer-token accuracy `0.4645 → 0.9551`; exact match `0.0000 → 0.5000`; 3-token prefix match `0.0000 → 0.5000` |
| Per-field generation accuracy (24 greedy outputs) | `sev=` `0.0 → 0.875`, `cat=` `0.0 → 0.5417`, `act=` `0.0 → 0.875` |
| Adapter export | 5 files copied, 540,672 params, tokenizer reloaded from the export (`Qwen2Tokenizer`, vocab 151,665, pad/eos 151,643) |
| Merged export | 494,032,768 params; `max_logit_delta` 0.000303 against tolerance 0.05, measured on a real held-out prompt before and after `merge_and_unload` |
| Serving | `/health`, `/generate`, `/metrics` verified live against the adapter on `mps`; malformed bodies rejected with 422, an over-long prompt with 400, unknown routes with 404 |
| Tracking | `loraflow mlflow-ui` serves the SQLite store; five finished runs (two `train`, three `evaluate`) listed through its REST API, with the adapter directory and each JSON report attached as an artifact |
| Tests | module-focused pytest suite; Ruff lint and format checks |
| CPU path | `configs/ci_smoke.yaml` on `sshleifer/tiny-gpt2`: `cpu` + `torch.float32`, 2 steps, `train_loss` 10.83, accuracy deltas `+0.0000` — plumbing verified, learning is not the point of that config |

**Provenance and reproducibility.** Every figure above was produced from the current tree. The 90-step
run was repeated into a second output directory (`configs/repro_check.yaml` → `artifacts/repro-run/`)
and the two runs agree exactly: `best_eval_loss` `0.1346336007118225`, `train_loss` `0.8617594838142395`,
`checkpoint-90` as the best checkpoint, a byte-identical `adapter_model.safetensors`
(sha256 `2652b6c2bb83f259…e2bf5d` for both), and an evaluation report whose `baseline`, `fine_tuned`,
`delta` and `field_accuracy` blocks compare equal. Only wall clock differed (`413.23 s` vs `342.01 s`).
The adapter and merged exports were then re-run from that second adapter into
`artifacts/repro-exported` and `artifacts/repro-merged`: same 540,672 adapter params in 5 files, same
494,032,768 merged params, same `max_logit_delta` `0.000303` against tolerance `0.05`, both exit 0.
Serving and the MLflow UI were checked against the current tree too. No external benchmark is
reproduced anywhere in this table.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core logic and tests, no torch needed
pip install -e ".[train,serve]"    # ML stages
```

```bash
python scripts/make_dataset.py                 # deterministic 480 train / 120 validation / 120 test JSONL
loraflow data-check --config configs/train.yaml
loraflow train      --config configs/train.yaml
loraflow evaluate   --config configs/train.yaml
loraflow export     --config configs/train.yaml              # adapter + tokenizer
loraflow export     --config configs/train.yaml --merged     # merged standalone weights
loraflow serve      --config configs/train.yaml --port 8125
loraflow mlflow-ui                                             # tracked runs and metrics
pytest
```

Three configs sit alongside the main one:

- `configs/smoke.yaml` — 2 optimizer steps on the real model, MPS, no tracking.
- `configs/ci_smoke.yaml` — `sshleifer/tiny-gpt2` on CPU, 2 steps, 24 examples, tracking off; this is what CI runs.
- `configs/repro_check.yaml` — identical to `train.yaml` except the output directories; used for the
  repeat run behind the reproducibility claim above.

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

## Repository layout

```text
LoRAFlow/
├── .github/workflows/ci.yml     lint -> unit matrix -> full pipeline smoke
├── configs/
│   ├── default.yaml             every supported key, commented
│   ├── train.yaml               the verified 0.5B run
│   ├── repro_check.yaml         that run again, into a second output directory
│   ├── smoke.yaml               2-step MPS validation
│   └── ci_smoke.yaml            tiny-gpt2 / CPU / 2 steps, tracking off
├── scripts/make_dataset.py      deterministic synthetic corpus generator
├── src/loraflow/
│   ├── config.py                pydantic model, extra="forbid"
│   ├── runtime.py               device/dtype resolution, seeding, JSON helpers
│   ├── data.py                  JSONL loading, prompt templating, splitting  (torch-free)
│   ├── metrics.py               normalization, exact/prefix/field, perplexity (torch-free)
│   ├── training.py              LoRA/TRL SFT stage, checkpoints, metadata
│   ├── evaluation.py            teacher-forced + greedy comparison
│   ├── tracking.py              MLflow tracker plus a no-op fallback
│   ├── export.py                adapter export, merged export with logit check
│   ├── api.py                   FastAPI app, Prometheus metrics, model service
│   └── cli.py                   data-check | train | evaluate | export | serve | mlflow-ui
├── tests/                       module-focused unit and integration tests
├── Dockerfile / docker-compose.yml
└── pyproject.toml
```

Heavy imports (`torch`, `transformers`, `trl`, `peft`) live inside the ML modules, so
`pip install -e ".[dev]"` alone runs the data, metrics, config, and CLI test suites.

## The task used for verification

`scripts/make_dataset.py` writes a synthetic farm-animal triage corpus: each record is a
short free-text observation note plus a three-field answer (`sev=… cat=… act=…`) derived from
fixed rules. The action vocabulary (`log_only`, `recheck_12h`, `flag_vet_now`) and the
`key=value` schema are not something a base 0.5B model produces from pretraining alone, which
is why the baseline scores 0.0 exact match and the fine-tuned adapter is genuinely different
rather than nominally so. Records are generated from fixed, disjoint seeds for train and
validation, so reruns reproduce the same 480/120 files byte for byte. Use real instruction-tuning JSONL with the configured instruction and output fields, plus an optional input field. Plain-text-only records are not supported for supervised instruction tuning.

## Configuration

`configs/default.yaml` documents every key. The ones that change behaviour:

- `data.prompt_template` — `{instruction}` plus `{input_block}`, which expands to an
  `### Input:` section only when a record carries an input value; `validation_size` carves a
  seeded split when no `validation_file` is given; `max_train_examples` caps the corpus.
- `training.device` / `training.dtype` — `auto` resolves CUDA → MPS → CPU and picks bf16 on CUDA,
  fp32 otherwise. An explicitly requested but unavailable device raises instead of silently moving.
- `training.warmup_steps`, `lr_scheduler`, `weight_decay`, `eval_steps`, `save_steps`,
  `gradient_checkpointing`, `max_steps`, `seed`, `run_name`.
- `evaluation.*` — `max_examples` (teacher-forced scoring), `generation_examples` (greedy decoding),
  `max_new_tokens`, `batch_size`; both models score the *same* held-out list.
- `tracking.*` — `sqlite:///mlflow.db` by default because MLflow 3.x keeps the plain filesystem
  store in maintenance mode; artifacts go to `artifacts_dir`.
- `export.*` — `output_dir`, `merged_dir`, `merge_and_save`, `save_dtype`, `verify_reload`.
- `serving.*` — `host`, `port`, `device`, `max_new_tokens`, `max_input_chars`,
  `max_concurrent_requests`, `log_level`.

## Inference API

`loraflow serve` loads the configured base model, applies the adapter when the directory exists,
and reports the result through `/health` (200 `ok`, 503 `degraded` while a load error is pending).

```bash
curl -s localhost:8125/generate -H 'content-type: application/json' -d '{
  "prompt": "Read the barn health note and label it with severity, category and action. Reply with exactly one line in the form sev=<severity> cat=<category> act=<action>. Use only the codes defined by the triage table.",
  "input": "Animal A213: lost hair along the left flank. Moving normally, with no other change today.",
  "max_new_tokens": 24
}'
# {"completion":"sev=routine cat=skin act=log_only", "completion_line":"sev=routine cat=skin act=log_only",
#  "prompt_tokens":78, "completion_tokens":11, "adapter_applied":true, "elapsed_seconds":0.3411}
```

`prompt` maps to the instruction field and `input` to the input field, so a request reproduces the
exact prompt the adapter was trained on (`format_prompt: false` sends raw text instead). Responses
carry both `completion` and `completion_line`. Send the training instruction verbatim: an off-distribution
instruction ("Triage this calf observation note.") was measured to return chat prose instead of the
`key=value` line, which is the adapter doing exactly what it was trained to do and nothing more.

Out-of-range bodies become 422 before any tensor is allocated (verified: empty `prompt`), an over-long
prompt becomes 400 with the limit named (verified: 5,046 characters against `max_input_chars: 4000`),
an unknown route becomes 404, and a model that failed to load becomes 503 with the reason rather than a
silent fallback. `/metrics` exposes `loraflow_requests_total`, `loraflow_generation_seconds`,
`loraflow_requests_in_flight`, and `loraflow_model_loaded`.

## Experiments

`train` and `evaluate` log the flattened config, LoRA hyperparameters, split statistics, step and
epoch counts, losses, the comparison deltas, and the artifact paths to MLflow, then attach
`training_metadata.json` / `evaluation_report.json` as artifacts. `loraflow mlflow-ui` opens the
local UI. `--no-track` (or `tracking.enabled: false`) swaps in a no-op tracker, and a missing MLflow
install logs a warning and degrades to no-op rather than failing the run.

The recorded experiment holds five finished runs: two training runs (61 params and 28 metrics each,
`train_loss` 0.8618, `best_eval_loss` 0.1346, artifacts `adapter/` + `training_metadata.json`) and three
evaluation runs (33 params and 36 `eval.*` metrics covering both models on the same 120 examples,
artifact `evaluation_report.json`). The default `tracking_uri` is a SQLite store because MLflow's `ui`
command takes `--backend-store-uri`, not `--tracking-uri` — the CLI passes the right one.

## Tests

`pytest` runs 154 test cases across the module-focused test files. Coverage: JSONL validation with the offending line
number, template rendering (empty input blocks, custom templates, whitespace-terminated templates,
unknown placeholders), seeded split determinism and disjointness, deduplication, split statistics,
device and dtype resolution including the unavailable-accelerator error, seeding, atomic JSON writes,
answer normalization, exact/prefix/per-field scoring, perplexity, masked token accuracy, delta
formatting, prompt/completion column shape, LoRA and `SFTConfig` translation (including
`completion_only_loss` and holding CUDA-only precision flags off the MPS path), best-checkpoint
selection, adapter parameter counting, answer-label masking and input-block recovery in evaluation,
export file/zero-byte/unknown-kind rejection and the logit-delta helper used by the merge check, every
API route including the 400/404/422/500/503 paths, and the CLI parser, its exit codes, and the
argument list `mlflow-ui` hands to MLflow's server.

Model-dependent suites gate on `pytest.importorskip`, so the core logic proves out without torch. The
actual base-versus-adapter comparison needs weights, so it is exercised by the CI pipeline job on
`sshleifer/tiny-gpt2` and by the recorded MPS run rather than as a unit test.

```bash
pytest                                        # everything installed
pytest tests/test_data.py tests/test_metrics.py  # core logic only
ruff check src tests scripts && ruff format --check src tests scripts
```

## Containers and CI

`docker build -t loraflow .` installs the CPU torch wheel on `python:3.11-slim`, then
`.[train,serve]`; the image health-checks `/health` and defaults to `loraflow serve`.
`docker compose up api` starts the server, `docker compose run --rm train` (profile `pipeline`, which
also covers `evaluate` and `export`) runs the stages against `configs/train.yaml` — that is the real
0.5B job, not a smoke test — and profile `tracking` adds an `mlflow/mlflow:3.18.0` UI pointed at the
same SQLite store and artifact directory the pipeline writes.

GitHub Actions runs lint, then the pure-logic suite on Python 3.10 and 3.13 with a dev-only install,
then a full pipeline job on the tiny model: dataset build, `data-check`, 2-step `train` with
artifact existence assertions, `evaluate` with report-completeness assertions, `export`, and a live
`serve` check that curls `/health`, `/generate`, `/metrics` and asserts the 422 path.

**Verification status of this section:** Docker is not installed on the development machine, so the
image was never built and the compose stack never started. CI validates the Compose configuration
syntax, but the image build and live container stack remain unverified. CI is configured to run the
full smoke pipeline; a completed green hosted run is required before claiming that path is verified.
Every command the pipeline job runs *was* executed locally from the current tree, in order, against
`configs/ci_smoke.yaml`: `make_dataset.py` (regenerated both files with byte-identical sha256 sums),
`data-check`, `train`, `evaluate`, and `export` — all exit 0, with `training_metadata.json`,
`evaluation_report.json` and `exported/adapter/export_manifest.json` on disk. On that config the tiny
random-init model is the point, not the accuracy: 2 steps on CPU, `train_loss` 10.83, and deltas of
`-0.0000` loss / `+0.0000` accuracy. The pipeline job proves plumbing; `configs/train.yaml` proves
learning.

## Hardware notes

Verified on an Apple M4 MacBook Air with 16 GB RAM, no CUDA. torch 2.15.0 MPS was checked directly
(matmul and bf16 both work) but training runs in fp32 on the Mac path, because bf16/fp16 training
coverage on MPS is not something to assume. The recorded 0.5B run completed in 413 s wall clock on a
single device and its merged export is a 1,976,163,472-byte `model.safetensors`. **QLoRA is not
implemented in this release**; the verified training path is standard LoRA. A separate CUDA-specific
quantized loading, training, and export path would be required before claiming QLoRA support.

## Known limitations

- Results come from a small synthetic task and a 0.5B model; they demonstrate the pipeline, not
  real-world quality. No public benchmark is reproduced.
- Training is single-process and single-device. `gradient_accumulation_steps` is the only
  parallelism knob; there is no FSDP, DeepSpeed, or multi-node support by design.
- Evaluation metrics are teacher-forced loss/accuracy plus greedy exact-match. They miss partial
  credit and semantic equivalence; there is no human evaluation and no calibration metric.
- The serving app holds one model per process with a bounded semaphore (`max_concurrent_requests`);
  it is a correctness-checked single-host server, not a throughput-tuned one. No auth, no rate
  limiting, no request queueing beyond the semaphore.
- MLflow tracking writes to a local SQLite store; the artifact location of an existing experiment is
  honoured rather than rewritten, so a pre-existing `loraflow` experiment keeps its original location
  and logs a warning.
- Merging is verified for bf16/fp32 saves. Lower-precision `save_dtype` values widen the tolerance to
  0.25 instead of proving equivalence.
- The container path is unverified: no Docker engine was available on the development machine, so
  `Dockerfile` and `docker-compose.yml` are reviewed configuration rather than tested artifacts.

## Reference

The high-level workflow was informed by the public project [Rishi625/LLM-Finetune-Pipeline](https://github.com/Rishi625/LLM-Finetune-Pipeline). LoRAFlow is a separate implementation; no source files are copied from that repository.
