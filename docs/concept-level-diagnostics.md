# Concept-level Prompt-v2 diagnostics

The aggregate category score does not show whether a weak category is failing on one held-out symptom concept or across every concept in that category. This utility groups the generated predictions already stored in each `evaluation_report.json` by the exact synthetic concept in the held-out input.

It reports, per concept:

- Example count.
- Category accuracy.
- Severity accuracy.
- Exact match across `sev`, `cat`, and `act`.
- Predicted category and severity counts, including missing or invalid outputs.

## Run against the Track C seed sweep

Run from the repository root after the three seed evaluations have completed:

```bash
python scripts/audit_v4_concept_performance.py \
  --reports \
  artifacts/v4-prompt-v2/seed-sweep/track-c/seed-42/evaluation_report.json \
  artifacts/v4-prompt-v2/seed-sweep/track-c/seed-43/evaluation_report.json \
  artifacts/v4-prompt-v2/seed-sweep/track-c/seed-44/evaluation_report.json \
  --output artifacts/v4-prompt-v2/seed-sweep/track-c-concept-diagnostics.json
```

The input reports must include generated `examples` with `input`, `gold`, and `fine_tuned` fields. The utility validates that each note contains exactly one known concept and that its gold category agrees with the concept inventory.

## Interpretation limits

This is a diagnostic breakdown of the existing fixed synthetic test set, not a new benchmark or an independent estimate of real-world generalization. Concepts have small and sometimes unequal sample counts. Compare patterns across seeds; do not rank concepts by a few examples or tune against this test set. The utility does not modify datasets, model weights, predictions, or evaluation metrics.
