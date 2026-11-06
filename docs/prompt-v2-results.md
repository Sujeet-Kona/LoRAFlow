# Prompt-v2 benchmark results

## Scope and protocol

Prompt-v2 is a rerun of the three controlled synthetic benchmark tracks. Each track has 600 training examples, 120 validation examples, and 150 held-out test examples. The reported exact-match and field metrics below come from the held-out test split, not the training or validation split.

- Base model: `Qwen/Qwen2.5-0.5B`
- Runtime: Apple Silicon MPS, `torch.float32`
- LoRA: rank 8, alpha 16, dropout 0.05, target modules `q_proj` and `v_proj`
- Trainable adapter parameters: 540,672 (0.1093% of 494,573,440 base parameters)
- Training: 3 epochs, 114 optimizer steps, seed 42
- Decoding: greedy, max 24 new tokens
- Test size: 150 examples per track

The TF-IDF + LinearSVC baseline fits its vocabulary and classifiers on training data only and is scored on the same held-out test examples. The action field is derived from predicted severity using the benchmark policy; it is not an independent classifier.

## Held-out results

| Metric | Track A | Track B | Track C |
| --- | ---: | ---: | ---: |
| LoRA exact match | 68.00% (102/150) | **77.33% (116/150)** | 37.33% (56/150) |
| TF-IDF + LinearSVC exact match | 30.00% (45/150) | 56.00% (84/150) | 16.67% (25/150) |
| Difference vs. classical baseline | +38.00 pp | +21.33 pp | +20.66 pp |
| Strict structured-output validity | 100.00% (150/150) | 98.67% (148/150) | 100.00% (150/150) |
| Answer-token accuracy | 97.77% | 98.36% | 95.23% |
| Mean answer loss | 0.054413 | 0.054019 | 0.109915 |
| Perplexity | 1.0559 | 1.0555 | 1.1162 |

Track B had two malformed outputs because the category value was the Chinese string `咳嗽`, which is outside the allowed label vocabulary. Do not round its structured-output validity to 100%.

The unadapted base Qwen model had 0% exact match on these runs because its generated prose frequently failed the one-line structured-output contract. This is a format-confounded baseline and is not a fair demonstration that LoRA improves on a capable base-model classifier. Use the classical baseline as the primary comparator, while still reporting the base-model result transparently.

## Field-level results

Values are accuracy / macro-F1. Action duplicates severity's scores because the action is deterministically derived from severity.

| Track | Severity | Category | Action (derived) |
| --- | ---: | ---: | ---: |
| A | 98.67% / 0.9608 | 68.67% / 0.7004 | 98.67% / 0.9608 |
| B | 90.67% / 0.6219 | 86.00% / 0.8606 | 90.67% / 0.6219 |
| C | 87.33% / 0.6043 | 42.67% / 0.4345 | 87.33% / 0.6043 |

Track C's category macro-F1 of 0.4345 is a clear weakness. Its per-class category recall is low for several classes, including behaviour (10%) and mobility (3.57%). Do not present Track C as a strong category classifier.

## Dataset and nearest-neighbour audit

The audit reports zero exact duplicate notes in each 150-example test split. A separate nearest-neighbour similarity audit sampled 60 test notes per track:

| Track | 1-NN exact match | Sampled test notes with train Jaccard >= 0.9 | Mean best Jaccard |
| --- | ---: | ---: | ---: |
| A | 29.33% | 11/60 (18.33%) | 0.8277 |
| B | 54.00% | 0/60 (0%) | 0.2445 |
| C | 20.00% | 0/60 (0%) | 0.1798 |

Track A therefore has no exact duplicate leakage, but the sampled near-duplicate signal is non-trivial and must be disclosed. Its unseen-concept framing is not a clean semantic generalization claim. Track B intentionally shares symptom concepts across splits, which helps explain why nearest-neighbour retrieval can recover category labels; the intended shift is severity-cue wording. Track C combines concept and severity wording shifts and is the most difficult condition.

These nearest-neighbour figures are from a 60-note audit sample, not all 150 test examples. The Jaccard threshold is a diagnostic, not a proof of semantic equivalence.

## Training and artifact provenance

| Track | Best checkpoint | Best eval loss | Runtime | MLflow training run ID |
| --- | --- | ---: | ---: | --- |
| A | checkpoint-114 | 0.0825923 | 735.7 s | `663d3fd037124c4d9f4674dd69a3778d` |
| B | checkpoint-114 | 0.0520588 | 787.6 s | `a3217410c7d54ff19a59c9f0ac8601de` |
| C | checkpoint-90 | 0.0986352 | 794.3 s | `d436421416bd45bb826adafaaea606c2` |

All three adapter export manifests report `reload_verified: true`, 540,672 adapter parameters, and the expected tokenizer files. This validates adapter export/reload, not live HTTP serving of every prompt-v2 adapter.

| Track | Config SHA-256 | Test split SHA-256 |
| --- | --- | --- |
| A | `e28c25289b362ad2f2b0a846a0ef985043fe14650d03c6b7706df05e81ace9e0` | `2e6c543afe7fcb03ee6a469c0b6772607780e7b8f4c5b1ccc9c54000f1bfc7dd` |
| B | `6de5f5e0f724b80e16ecf9454ef8d9d9a000d4a02d441875c3dcd783da3abbda` | `d0c6bde7b1e3698bbba7c6d04cf78ad7021d2898c42d9acaa174dcde1ac6c506` |
| C | `9aff794f901721737b89c80bb2ed8318137b235549347ef2b735d1aa2fd589f1` | `447488278ebc4f38627ecbf074abc2362cd71d991fe947e4315a7545a8e5fdd9` |

The exact prompt-v2 input JSONL files, per-track manifests, and three training/evaluation/export configs are now committed under `data/v4-prompt-v2/` and `configs/v4_prompt_v2_track_*.yaml`. The manifests record the SHA-256 digest and expected record count for each split. The exact corpus is treated as a versioned benchmark fixture; this repository does not claim that the original one-off data-generation procedure can regenerate byte-identical files.

## Clean-checkout reproduction

Use a fresh checkout with the committed benchmark inputs. The runner verifies every JSONL file against its manifest SHA-256 and record count, runs the pipeline's dataset validation, then trains, evaluates, exports, reload-validates, and live-smoke-tests all three adapters in sequence.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,serve,dev]"
bash -n scripts/run_prompt_v2_experiments.sh
bash scripts/run_prompt_v2_experiments.sh
```

The full run downloads `Qwen/Qwen2.5-0.5B` from Hugging Face and requires enough local memory and disk for the model, checkpoints, MLflow artifacts, and three adapter exports. On the original Apple M4 run, training alone took approximately 12–13 minutes per track. The runner refuses to overwrite existing run/export directories; use a fresh checkout or move prior artifacts aside intentionally. It is a reproduction of training/evaluation from the exact committed inputs, not a data-regeneration command.

## Engineering interpretation

- Track B is the headline prompt-v2 result: 77.33% exact match and 98.67% strict structured validity against a 56.00% train-only classical baseline.
- Track A's 68.00% exact match is promising as a pipeline result, but the near-duplicate audit caveat prevents a strong unseen-concept generalization claim.
- Track C's 37.33% exact match and 0.4345 category macro-F1 show the combined shift remains difficult.
- Preserve the original V4 results separately; do not replace them with prompt-v2 numbers without a same-protocol, same-split comparison.
- Do not use the base model's zero exact-match score as the headline improvement baseline.
- Do not describe the synthetic benchmark as clinical/veterinary validation or as a deployed decision system.

## Verification status and remaining limitations

- The exact input/config files are committed, and `scripts/run_prompt_v2_experiments.sh` provides clean-checkout training/evaluation/export reproduction with hash and count verification.
- The live FastAPI smoke test passed for all three exported Prompt-v2 adapters: health checks succeeded, each reported `adapter_applied: true`, and each returned a schema-valid generation.
- A regression test now ensures out-of-vocabulary structured labels are counted as invalid rather than accepted as valid labels.
- Re-run the full runner only on a fresh checkout; it is intentionally not safe to overwrite the existing experiment artifacts.
- GitHub Actions on the latest branch commit must pass before the PR is considered CI-verified.

No Docker image build or live Compose deployment is claimed or required for this project scope.
