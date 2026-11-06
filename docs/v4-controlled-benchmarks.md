# LoRAFlow v4 controlled benchmark design

v4 is additive: it never rewrites v1, v2, or v3. Generated files live under `data/v4/` and are ignored by Git.

| Track | Symptom concepts | Severity wording | Purpose |
|---|---|---|---|
| A | Disjoint (5 train, 1 validation, 2 test concepts/category) | Shared across splits | Isolate unseen symptom-concept recognition |
| B | All 48 concepts shared across default splits; a category-balanced common subset for smaller smoke splits | Disjoint by split | Isolate unseen severity-cue wording |
| C | Disjoint (same allocation as A) | Disjoint by split | Measure combined distribution shift |

The generator removes random animal IDs. Six categories have eight curated concepts each. Default split sizes are 600/120/150 per track. Each track has its own deterministic seed, JSONL files, manifest, audit summary, and SHA-256 hashes.

## Controlled severity sampling

The original retry-until-unique sampler could distort severity proportions differently when tracks had different concept inventories. v4 now builds the finite scenario pool, guarantees coverage of every concept allocated to a split, and samples unique combinations without replacement to exact severity quotas.

Target ratios are 7.5% routine, 37% watch, and the remainder urgent. For the default split sizes, the exact targets are:

| Split size | Routine | Watch | Urgent |
|---:|---:|---:|---:|
| 600 | 45 | 222 | 333 |
| 120 | 9 | 44 | 67 |
| 150 | 11 | 56 | 83 |

The small integer rounding difference at 120/150 rows is recorded in each manifest. For reduced smoke splits smaller than the 48-concept inventory, Track B uses the same deterministic category-balanced subset in every split. These are controlled benchmark sampling proportions, not estimates of real-world prevalence.

## Generate

```bash
python scripts/make_v4_benchmarks.py
python scripts/make_v4_benchmarks.py --train 60 --validation 24 --test 24 --out-dir /tmp/loraflow-v4-smoke
```

After the full datasets have been generated and audited, install the training and development extras and run the sequential experiment driver:

```bash
pip install -e ".[train,dev]"
bash scripts/run_v4_experiments.sh
```

The driver does not regenerate or modify datasets. It reuses a complete adapter already present (including the completed Track A adapter), trains only missing tracks, evaluates A/B/C one at a time, saves per-stage logs under `artifacts/v4-logs/`, and writes `artifacts/v4-summary.md`. It stops on the first error so incomplete results are not presented as a completed comparison.

The generator checks exact concept coverage, concept-to-category label consistency, split concept allocation, exact severity quotas, valid target shape, severity-to-action policy, cue separation, and cross-split input uniqueness before writing each manifest. Each JSONL file is SHA-256 hashed.

## Controlled experiment protocol

1. Train a separate adapter/config/output directory for each track.
2. Tune only on train and validation. Never select checkpoints, thresholds, or prompts based on test.
3. Report per-field accuracy and macro-F1, per-class precision/recall, confusion matrices, exact match, strict structured-output validity, malformed outputs, and invalid values.
4. Action is deterministically derived from severity; report it, but do not describe it as independent evidence.
5. Compare against the included TF-IDF + LinearSVC baseline for category and severity. It uses raw observation text only, fits its vocabulary and classifiers on train only, and uses fixed hyperparameters (word uni/bi-grams, sublinear TF, class-balanced LinearSVC, C=1.0); no test-driven tuning is performed. If hyperparameters are changed, select them using validation only and record that protocol.
6. Score the classical baseline on the same complete held-out test split. Its action output is derived from predicted severity using the benchmark policy and is not an independent classifier.
7. Save predictions, configs, seeds, environment, split hashes, and adapter identifiers for reproducibility.

`loraflow evaluate` now reports three distinct comparisons when the data contains `sev` and `cat` labels: the unadapted base LLM, the LoRA adapter, and the train-only TF-IDF + LinearSVC baseline. The classical baseline is reported separately as `classical_baseline`; it does not replace the base-model comparison.

## Limitations

This is a curated synthetic benchmark with a small concept inventory. It helps diagnose controlled shifts in this pipeline; it does not establish broad natural-language generalization, veterinary utility, or clinical safety. Do not present v3's 29.17% exact-match score as a v4 result.
