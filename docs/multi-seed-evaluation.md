# Prompt-v2 multi-seed evaluation

The published Prompt-v2 results use seed 42. A single seed is not enough to show whether a result is stable to training randomness. This runner repeats a selected track with multiple training seeds while holding the committed train/validation/test files, model, hyperparameters, prompt template, and greedy decoding protocol fixed.

## Run the weak combined-shift track

The default is Track C and seeds 42, 43, and 44:

```bash
pip install -e ".[train,serve,dev]"
python scripts/run_prompt_v2_seed_sweep.py --track c --seeds 42 43 44
```

Track C is the priority because the existing single-seed result has 37.33% exact match and 0.4345 category macro-F1. The sweep retrains and evaluates each seed sequentially, exports each adapter with reload verification, and writes a machine-readable summary to `artifacts/v4-prompt-v2/seed-sweep/track-c-summary.json`.

To inspect the strongest track instead:

```bash
python scripts/run_prompt_v2_seed_sweep.py --track b --seeds 42 43 44
```

The runner verifies the committed benchmark manifests before training and refuses to overwrite any existing seed output. Move the output directory aside or choose a new `--output-root` to rerun intentionally.

## Reported metrics

For each seed and as mean, sample standard deviation, minimum, and maximum across seeds, the summary reports:

- Generated exact match and prefix match.
- Answer-token accuracy, mean answer loss, and perplexity.
- Category and severity macro-F1.
- Strict structured-output validity.
- Train-only TF-IDF + LinearSVC exact match for context.

The same fixed held-out examples are used for every seed. The seed-level standard deviation is a descriptive measure of training-seed sensitivity; the script deliberately does not report a confidence interval or treat examples as independent training runs. The test set is not used for checkpoint selection or hyperparameter tuning.

This remains a small synthetic benchmark. Multi-seed stability does not establish broad natural-language generalization, veterinary utility, or clinical safety. The sweep is a local research workflow and is not run in CI because it downloads model weights and performs multiple full training runs.

## Dataset label-consistency audit

The v4 benchmark audit now reconstructs severity from the three explicit input cue dimensions (systemic condition, functional ability, and trajectory) using the benchmark's declared rule, then checks that the target severity agrees. It requires exactly one cue from each dimension and uses the correct split-specific wording (Track A intentionally shares the training wording). Regression tests cover a corrupted severity target, a missing cue, and valid split-specific wording.

This is a **generator-rule consistency check**, not an independent semantic or clinical assessment: the same declared rule defines both the generated target and the audit expectation. It can catch corrupted labels and malformed cue construction, but it cannot prove that the rule itself is a valid interpretation of an animal observation. The stratified severity quotas also intentionally alter the natural frequency of routine/watch/urgent cases; the benchmark is not a prevalence estimate.
