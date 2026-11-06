"""Generate isolated v4 controlled-generalization benchmark tracks.

Track A isolates unseen symptom concepts; Track B isolates unseen severity wording;
Track C combines both shifts. Synthetic engineering data only, never clinical guidance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Classify the observation using only these exact labels: "
    "severity must be routine, watch, or urgent; "
    "category must be mobility, feeding, respiratory, skin, behaviour, or unknown; "
    "action must be log_only, recheck_12h, or flag_vet_now. "
    "The action mapping is routine=log_only, watch=recheck_12h, urgent=flag_vet_now. "
    "Return exactly one line in this format: "
    "sev=<severity> cat=<category> act=<action>. Do not add explanations or other text."
)
CATEGORIES = ("mobility", "feeding", "respiratory", "skin", "behaviour", "unknown")
SEVERITIES = ("routine", "watch", "urgent")
ACTIONS = {"routine": "log_only", "watch": "recheck_12h", "urgent": "flag_vet_now"}
CONCEPTS = {
    "mobility": [
        "uneven weight bearing",
        "reluctance to use a front leg",
        "slow rising after rest",
        "shortened steps",
        "repeated stumbling",
        "stiffness after standing",
        "difficulty turning",
        "reduced walking distance",
    ],
    "feeding": [
        "reduced feed intake",
        "lower water consumption",
        "difficulty chewing",
        "slow eating",
        "gradual body condition loss",
        "refusal of usual feed",
        "frequent dropping of feed",
        "unusual feeding pauses",
    ],
    "respiratory": [
        "repeated coughing",
        "nasal discharge",
        "rapid breathing at rest",
        "noisy breathing",
        "increased breathing effort",
        "persistent sneezing",
        "shallow breaths",
        "breathing pauses",
    ],
    "skin": [
        "irritated patch beneath the coat",
        "repeated rubbing against a post",
        "small open skin area",
        "hair loss along one side",
        "swelling near a wound",
        "scaly skin region",
        "localized redness",
        "repeated scratching",
    ],
    "behaviour": [
        "withdrawal from the group",
        "failure to respond to a familiar call",
        "repeated head pressing",
        "restlessness during handling",
        "change in usual routine",
        "unusual isolation",
        "reduced interaction",
        "repetitive pacing",
    ],
    "unknown": [
        "unexplained change in condition",
        "observation cannot be localised",
        "illness report without detail",
        "change observer cannot describe",
        "no clear symptom recorded",
        "general concern without specifics",
        "uncertain reported change",
        "observation lacks a clear description",
    ],
}
CUES = {
    "train": {
        "normal": "general checks remain within the usual range",
        "mild": "a slight overall change is present",
        "critical": "the animal is markedly dull with a very high temperature",
        "limited": "some usual activity is difficult",
        "failed": "a basic usual activity cannot be completed",
        "improving": "the change has been easing",
        "stable": "the observation has remained much the same",
        "worsening": "the change has become more pronounced",
    },
    "validation": {
        "normal": "the wider assessment appears unremarkable",
        "mild": "the overall condition is somewhat reduced",
        "critical": "the animal is difficult to rouse and feels extremely hot",
        "limited": "only part of the usual activity is possible",
        "failed": "a basic activity can no longer be completed",
        "improving": "the signs have started to settle",
        "stable": "the pattern has not shifted",
        "worsening": "the signs are becoming more apparent",
    },
    "test": {
        "normal": "no broader abnormality was noticed",
        "mild": "the wider check suggests a modest change",
        "critical": "the animal barely responds and is extremely hot",
        "limited": "ordinary activity is noticeably restricted",
        "failed": "an essential activity is no longer possible",
        "improving": "there has been a gradual recovery",
        "stable": "the findings are still at the previous level",
        "worsening": "the findings have intensified",
    },
}

# Stratified targets prevent rejection sampling from changing label proportions
# as the number of available symptom concepts changes between tracks/splits.
SEVERITY_TARGET_RATIOS = {"routine": 0.075, "watch": 0.37}


def derive_severity(systemic, function, trajectory):
    if systemic == "critical" or function == "failed":
        return "urgent"
    if systemic == "mild" or function == "limited" or trajectory == "worsening":
        return "watch"
    return "routine"


def target_severity_counts(count):
    routine = round(count * SEVERITY_TARGET_RATIOS["routine"])
    watch = round(count * SEVERITY_TARGET_RATIOS["watch"])
    return {"routine": routine, "watch": watch, "urgent": count - routine - watch}


def infer_severity_from_note(note, split):
    """Reconstruct severity from the split's actual cue phrases.

    This checks label-to-input consistency against the benchmark's declared
    rule. It is not an independent clinical or semantic judgement.
    """
    cue_split = "train" if split == "train" else split
    cues = CUES[cue_split]
    dimensions = {
        "systemic": ("normal", "mild", "critical"),
        "function": ("normal", "limited", "failed"),
        "trajectory": ("improving", "stable", "worsening"),
    }
    observed = {}
    for dimension, keys in dimensions.items():
        matches = [key for key in keys if cues[key] in note]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {dimension} cue for {split}, found {matches}"
            )
        observed[dimension] = matches[0]
    return derive_severity(
        observed["systemic"], observed["function"], observed["trajectory"]
    )


def allowed_concepts(track, split, category):
    if track == "B":
        return CONCEPTS[category]
    if split == "train":
        return CONCEPTS[category][:5]
    if split == "validation":
        return CONCEPTS[category][5:6]
    if split == "test":
        return CONCEPTS[category][6:]
    raise ValueError(f"Unknown split: {split}")


def _concepts_for_split(track, split, count, shared_concept_count=None):
    if track != "B":
        concepts = [
            phrase for category in CATEGORIES for phrase in allowed_concepts(track, split, category)
        ]
        if count < len(concepts):
            raise ValueError(
                f"{track}/{split} needs at least {len(concepts)} rows to cover its concepts"
            )
        return concepts

    # Use one split-independent concept budget so every split shares the same
    # category-balanced inventory, including reduced smoke datasets.
    inventory = [
        CONCEPTS[category][index]
        for index in range(len(CONCEPTS[CATEGORIES[0]]))
        for category in CATEGORIES
    ]
    concept_count = (
        len(inventory)
        if shared_concept_count is None
        else min(shared_concept_count, len(inventory))
    )
    return inventory[: min(count, concept_count)]


def _scenario_candidates(track, split, count, shared_concept_count=None):
    cue_split = "train" if track == "A" else split
    cues = CUES[cue_split]
    candidates = {severity: [] for severity in SEVERITIES}
    by_concept = {}
    selected_concepts = set(_concepts_for_split(track, split, count, shared_concept_count))
    for category in CATEGORIES:
        for concept in allowed_concepts(track, split, category):
            if concept not in selected_concepts:
                continue
            by_concept[concept] = {severity: [] for severity in SEVERITIES}
            for systemic, function, trajectory in itertools.product(
                ("normal", "mild", "critical"),
                ("normal", "limited", "failed"),
                ("improving", "stable", "worsening"),
            ):
                severity = derive_severity(systemic, function, trajectory)
                note = (
                    f"Observation: {concept}. {cues[systemic]}. "
                    f"{cues[function]}. {cues[trajectory]}."
                )
                row = {
                    "instruction": INSTRUCTION,
                    "input": note,
                    "output": f"sev={severity} cat={category} act={ACTIONS[severity]}",
                }
                candidates[severity].append(row)
                by_concept[concept][severity].append(row)
    return candidates, by_concept


def generate(track, split, count, seed, shared_concept_count=None):
    if track not in ("A", "B", "C"):
        raise ValueError(f"Unknown track: {track}")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    if count < 1:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    candidates, by_concept = _scenario_candidates(track, split, count, shared_concept_count)
    targets = target_severity_counts(count)
    selected = []
    selected_notes = set()
    remaining = dict(targets)

    # Guarantee every allocated symptom concept is represented while consuming
    # the class quotas proportionally; routine is used only as needed for coverage.
    concepts = list(by_concept)
    rng.shuffle(concepts)
    for concept in concepts:
        eligible = [
            severity
            for severity in SEVERITIES
            if remaining[severity] > 0 and by_concept[concept][severity]
        ]
        if not eligible:
            raise ValueError("Requested split is too small to cover its concept inventory")
        severity = max(eligible, key=lambda item: remaining[item])
        row = rng.choice(by_concept[concept][severity])
        selected.append(row)
        selected_notes.add(row["input"])
        remaining[severity] -= 1

    # Fill exact severity quotas by sampling unique scenario combinations without
    # replacement. This avoids the label drift caused by retry-until-unique sampling.
    for severity in SEVERITIES:
        pool = [row for row in candidates[severity] if row["input"] not in selected_notes]
        rng.shuffle(pool)
        needed = remaining[severity]
        if needed > len(pool):
            raise ValueError(
                f"Insufficient unique {severity} scenarios: need {needed}, have {len(pool)}"
            )
        additions = pool[:needed]
        selected.extend(additions)
        selected_notes.update(row["input"] for row in additions)

    if len(selected) != count or len(selected_notes) != count:
        raise AssertionError("Generator failed to produce the requested unique row count")
    rng.shuffle(selected)
    return selected


def audit(track, data):
    shared_concept_count = min(48, min(map(len, data.values()))) if track == "B" else None
    concepts = {}
    severity_counts = {}
    for split, rows in data.items():
        found = set()
        counts = {severity: 0 for severity in SEVERITIES}
        for row in rows:
            matches = [
                phrase
                for values in CONCEPTS.values()
                for phrase in values
                if phrase in row["input"]
            ]
            if len(matches) != 1:
                raise ValueError(f"{track}/{split}: expected one symptom concept, found {matches}")
            found.add(matches[0])
            parts = row["output"].split()
            if len(parts) != 3 or not all("=" in part for part in parts):
                raise ValueError("Malformed target")
            severity, category, action = (part.split("=", 1)[1] for part in parts)
            if severity not in ACTIONS or category not in CATEGORIES:
                raise ValueError("Unknown target label")
            inferred_severity = infer_severity_from_note(row["input"], split)
            if severity != inferred_severity:
                raise ValueError(
                    f"{track}/{split}: severity label {severity!r} conflicts "
                    f"with input-derived severity {inferred_severity!r}"
                )
            concept_category = next(
                category_name
                for category_name, phrases in CONCEPTS.items()
                if matches[0] in phrases
            )
            if category != concept_category:
                raise ValueError(
                    f"{track}/{split}: category label {category!r} does not "
                    f"match concept category {concept_category!r}"
                )
            if ACTIONS[severity] != action:
                raise ValueError("Target action does not match severity")
            counts[severity] += 1

        expected = set(_concepts_for_split(track, split, len(rows), shared_concept_count))
        if found != expected:
            raise ValueError(
                f"{track}/{split}: concept coverage mismatch; "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        target_counts = target_severity_counts(len(rows))
        if counts != target_counts:
            raise ValueError(
                f"{track}/{split}: severity counts {counts} != targets {target_counts}"
            )
        concepts[split] = found
        severity_counts[split] = counts

    if track in ("A", "C"):
        if (
            concepts["train"] & concepts["validation"]
            or concepts["train"] & concepts["test"]
            or concepts["validation"] & concepts["test"]
        ):
            raise ValueError("Held-out symptom concepts overlap")
    else:
        shared = set.intersection(*(set(values) for values in concepts.values()))
        if len(shared) != min(len(values) for values in concepts.values()):
            raise ValueError("Track B must preserve every concept from the smallest split")

    cue_sets = {
        split: {
            CUES[("train" if track == "A" else split)][key]
            for key in (
                "normal",
                "mild",
                "critical",
                "limited",
                "failed",
                "improving",
                "stable",
                "worsening",
            )
        }
        for split in data
    }
    if track == "A":
        if not (cue_sets["train"] == cue_sets["validation"] == cue_sets["test"]):
            raise ValueError("Track A must share severity wording")
    elif (
        cue_sets["train"] & cue_sets["validation"]
        or cue_sets["train"] & cue_sets["test"]
        or cue_sets["validation"] & cue_sets["test"]
    ):
        raise ValueError("Severity cue wording overlaps across splits")

    all_notes = [row["input"] for rows in data.values() for row in rows]
    if len(all_notes) != len(set(all_notes)):
        raise ValueError("Duplicate input notes across splits")

    return {
        "counts": {split: len(rows) for split, rows in data.items()},
        "concepts": {split: sorted(values) for split, values in concepts.items()},
        "severity_counts": severity_counts,
        "severity_targets": {
            split: target_severity_counts(len(rows)) for split, rows in data.items()
        },
        "severity_cue_sets": {split: sorted(values) for split, values in cue_sets.items()},
        "policy_checks": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/v4"))
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--validation", type=int, default=120)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1:
        parser.error("split sizes must be positive")

    for offset, track in enumerate(("A", "B", "C")):
        shared_concept_count = (
            min(48, args.train, args.validation, args.test) if track == "B" else None
        )
        data = {
            split: generate(
                track,
                split,
                size,
                args.seed + offset * 100 + index,
                shared_concept_count,
            )
            for index, (split, size) in enumerate(
                (
                    ("train", args.train),
                    ("validation", args.validation),
                    ("test", args.test),
                )
            )
        }
        report = audit(track, data)
        destination = args.out_dir / f"track_{track.lower()}"
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for split, rows in data.items():
            path = destination / f"{split}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "dataset_version": "v4",
            "track": track,
            "seed": args.seed + offset * 100,
            "counts": report["counts"],
            "audit": report,
            "concept_inventory": CONCEPTS,
            "concept_policy": (
                "B shares one category-balanced concept inventory across splits; "
                "A/C hold out concepts "
                "(5 train, 1 validation, 2 test concepts per category)."
            ),
            "severity_cue_policy": (
                "A shares severity wording; B/C use split-specific severity wording."
            ),
            "severity_sampling_policy": {
                "target_ratios": SEVERITY_TARGET_RATIOS,
                "remaining_class": "urgent",
                "description": (
                    "Exact stratified class quotas; unique scenario combinations sampled "
                    "without replacement after ensuring every allocated concept appears."
                ),
            },
            "severity_policy": {
                "urgent": "systemic=critical OR function=failed",
                "watch": ("otherwise systemic=mild OR function=limited OR trajectory=worsening"),
                "routine": "otherwise",
            },
            "action_by_severity": ACTIONS,
            "sha256": hashes,
            "synthetic_data_warning": (
                "Engineering benchmark only; not veterinary guidance or clinical decision support."
            ),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"Track {track}: {report['counts']} | "
            f"severity={report['severity_counts']} | audit passed | {destination}"
        )


if __name__ == "__main__":
    main()
