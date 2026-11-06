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

## Completed Track C sweep (seeds 42, 43, 44)

The three-seed sweep was executed locally on Apple Silicon MPS with `torch.float32`. All three runs used the same committed 150-example test split (SHA-256 recorded in the summary), greedy decoding, and the same training/evaluation configuration.

| Metric | Mean | Sample std. dev. | Min–max |
| --- | ---: | ---: | ---: |
| Generated exact match | 48.44% | 10.70 pp | 37.33–58.67% |
| Category macro-F1 | 53.83% | 10.43 pp | 43.45–64.30% |
| Severity macro-F1 | 61.15% | 1.12 pp | 60.43–62.44% |
| Answer-token accuracy | 96.06% | 0.82 pp | 95.23–96.86% |
| Strict structured validity | 99.78% | 0.39 pp | 99.33–100.00% |
| Mean answer loss | 0.09798 | 0.01161 | 0.08673–0.10992 |
| Perplexity | 1.1030 | 0.01282 | 1.0906–1.1162 |

The train-only TF-IDF + LinearSVC baseline achieved 16.67% exact match on the same fixed test set in all three reports. The LoRA exact-match mean is therefore 31.78 percentage points higher on this synthetic benchmark. This is a controlled benchmark comparison, not evidence of real-world or broad natural-language generalization. The unadapted Qwen baseline's 0% exact match is format-confounded and is not used as the headline comparator.

### Interpretation and failure analysis

- Exact match varies materially by training seed (37.33–58.67%). The 10.70-point sample standard deviation is evidence of seed sensitivity, not a confidence interval.
- Category macro-F1 is also seed-sensitive (43.45–64.30%). The held-out mobility class is a recurring weakness; the per-seed reports should be consulted rather than hiding this behind aggregate accuracy.
- Severity macro-F1 is more stable, but the rare routine class has only 11 test examples. In the inspected seed-43 and seed-44 reports, routine recall is zero. This is a serious per-class failure despite high overall severity accuracy.
- Action is a deterministic mapping from severity and is explicitly supplied in the task instruction. Its score is not independent evidence of a second learned classification capability.
- The benchmark's severity quotas are intentionally stratified at approximately 7.5% routine, 37% watch, and 55.5% urgent. The small routine support is a property of the benchmark design, not an accidental split imbalance.
- Track C intentionally combines unseen symptom concepts with split-specific severity wording. Weakness on this track is the experiment's finding; it should not be “fixed” by changing the held-out test set or tuning against it.

These results support claims about implementing a repeatable LoRA experimentation/evaluation pipeline. They do not support claims that the model reliably classifies all categories or severity classes, and they do not establish veterinary utility.


This remains a small synthetic benchmark. Multi-seed stability does not establish broad natural-language generalization, veterinary utility, or clinical safety. The sweep is a local research workflow and is not run in CI because it downloads model weights and performs multiple full training runs.
