"""Audit a training split against one or more held-out splits.

Reports size, schema, label distributions, duplication, template leakage and a pure
memorization baseline (1-nearest-neighbour lookup over the training notes) for each held
split, so a fine-tuned score can be compared against what pattern reuse alone achieves.

Usage:
    python scripts/audit_dataset.py --train data/train.jsonl \
        --held validation=data/validation.jsonl \
        --output artifacts/audit/train_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loraflow.audit import audit_records, read_records  # noqa: E402


def _parse_pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--held expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def _validate_manifest(manifest_path: Path, splits: dict[str, list[dict]]) -> dict:
    """Verify that held-out symptom concepts are genuinely absent from training."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    concepts = manifest["symptom_concepts_by_split"]
    required = {"train", "validation", "test"}
    if set(concepts) != required or set(splits) != required:
        raise ValueError("manifest and audit must contain train, validation, and test splits")

    phrase_sets = {
        name: {phrase for phrases in categories.values() for phrase in phrases}
        for name, categories in concepts.items()
    }
    overlaps = {
        "train_validation": sorted(phrase_sets["train"] & phrase_sets["validation"]),
        "train_test": sorted(phrase_sets["train"] & phrase_sets["test"]),
        "validation_test": sorted(phrase_sets["validation"] & phrase_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"symptom concepts overlap across splits: {overlaps}")

    action_map = manifest["action_by_severity"]
    cue_maps = manifest["severity_cues_by_split"]
    checked = {}
    for split_name, records in splits.items():
        expected_count = manifest["counts"][split_name]
        if len(records) != expected_count:
            raise ValueError(
                f"{split_name} count mismatch: manifest={expected_count}, actual={len(records)}"
            )
        for index, record in enumerate(records, start=1):
            text = str(record.get("input", ""))
            matches = [
                (category, phrase)
                for category, phrases in concepts[split_name].items()
                for phrase in phrases
                if phrase in text
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{split_name} record {index}: expected exactly one known symptom concept, "
                    f"found {len(matches)}"
                )
            expected_category, _ = matches[0]
            labels = dict(re.findall(r"(sev|cat|act)=([a-z0-9_]+)", str(record.get("output", ""))))
            if set(labels) != {"sev", "cat", "act"}:
                raise ValueError(f"{split_name} record {index}: malformed output labels {labels}")
            if labels["cat"] != expected_category:
                raise ValueError(
                    f"{split_name} record {index}: category {labels['cat']} does not match "
                    f"symptom concept category {expected_category}"
                )
            observed_states = {}
            for dimension, state_cues in cue_maps[split_name].items():
                matched_states = [
                    state
                    for state, phrases in state_cues.items()
                    if any(phrase in text for phrase in phrases)
                ]
                if len(matched_states) != 1:
                    raise ValueError(
                        f"{split_name} record {index}: expected one {dimension} cue, "
                        f"found {matched_states}"
                    )
                observed_states[dimension] = matched_states[0]
            if (
                observed_states["systemic"] == "critical"
                or observed_states["function"] == "failed"
            ):
                expected_severity = "urgent"
            elif (
                observed_states["systemic"] == "mild"
                or observed_states["function"] == "limited"
                or observed_states["trajectory"] == "worsening"
            ):
                expected_severity = "watch"
            else:
                expected_severity = "routine"
            if labels["sev"] != expected_severity:
                raise ValueError(
                    f"{split_name} record {index}: severity {labels['sev']} does not match "
                    f"policy result {expected_severity}"
                )
            if labels["sev"] not in action_map or labels["act"] != action_map[labels["sev"]]:
                raise ValueError(
                    f"{split_name} record {index}: action {labels['act']} is inconsistent "
                    f"with severity {labels['sev']}"
                )
        checked[split_name] = len(records)

    return {
        "dataset_version": manifest.get("dataset_version"),
        "records_checked": checked,
        "symptom_concept_counts": {
            name: sum(len(phrases) for phrases in categories.values())
            for name, categories in concepts.items()
        },
        "cross_split_symptom_overlap": overlaps,
        "category_consistency": "passed",
        "severity_policy_consistency": "passed",
        "action_by_severity_consistency": "passed",
    }


def _summary(audit: dict) -> dict:
    """The handful of numbers worth reading first."""

    train = audit["train"]
    return {
        "train_size": train["size"],
        "train_distinct_instructions": train["instructions"]["distinct_instructions"],
        "train_distinct_output_strings": train["labels"]["unique_output_strings"],
        "train_distinct_skeletons": train["templates"]["distinct_skeletons"],
        "train_shapes_that_fix_the_label": train["templates"]["shapes_that_fix_the_label"],
        "held": {
            name: {
                "size": block["size"],
                "exact_shared_notes": block["overlap_with_train"]["exact_shared_notes"],
                "template_leakage": block["overlap_with_train"]["template_leakage"],
                "novel_words": block["overlap_with_train"]["novel_words_in_held"],
                "held_within_0_9_of_train": block["near_duplicates_vs_train"][
                    "held_within_0_9_of_a_train_note"
                ],
                "memorize_exact_match": block["memorization_baseline"]["exact_match_1nn"],
                "memorize_per_field": block["memorization_baseline"]["per_field_1nn"],
                "majority_per_field": block["memorization_baseline"]["per_field_majority"],
            }
            for name, block in audit["held"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--held",
        type=_parse_pair,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Held-out split to audit; repeat for several splits",
    )
    parser.add_argument("--output", type=Path, help="Write the full audit JSON here")
    parser.add_argument(
        "--manifest", type=Path, help="Validate the generator contract and concept holdouts"
    )
    args = parser.parse_args()

    train = read_records(args.train)
    paths = dict(args.held)
    held = {name: read_records(path) for name, path in paths.items()}
    audit = audit_records(train, held)
    if args.manifest:
        audit["generator_contract"] = _validate_manifest(args.manifest, {"train": train, **held})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "files": {"train": str(args.train), **{k: str(v) for k, v in paths.items()}},
                "summary": _summary(audit),
                "generator_contract": audit.get("generator_contract"),
                "full_audit": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
