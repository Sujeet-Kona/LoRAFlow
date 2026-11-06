"""Generate a deterministic, compositional demo corpus for LoRAFlow.

This synthetic animal-observation task is not veterinary guidance. Category is derived
from the symptom concept; severity is derived from independent systemic, functional,
and trajectory attributes. The output action follows an explicit policy mapping.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "Read the animal observation and return its severity, category and next action. "
    "Reply with exactly one line: sev=<severity> cat=<category> act=<action>. "
    "Use only the allowed codes."
)

CATEGORIES = ("mobility", "feeding", "respiratory", "skin", "behaviour", "unknown")
SEVERITIES = ("routine", "watch", "urgent")
ACTION_BY_SEVERITY = {
    "routine": "log_only",
    "watch": "recheck_12h",
    "urgent": "flag_vet_now",
}

SYMPTOMS: dict[str, tuple[str, ...]] = {
    "mobility": (
        "uneven weight-bearing while walking",
        "reluctance to place weight on a front hoof",
        "slow rising after rest",
        "shortened steps compared with nearby animals",
        "repeated stumbling on level ground",
    ),
    "feeding": (
        "reduced feed intake",
        "very small amounts of feed consumed at a meal",
        "lower water consumption than usual",
        "difficulty chewing and retaining feed",
        "gradual loss of body condition",
    ),
    "respiratory": (
        "repeated coughing",
        "nasal discharge",
        "rapid breathing while resting",
        "unusual sounds during breathing",
        "increased effort during breathing",
    ),
    "skin": (
        "an irritated patch beneath the coat",
        "repeated rubbing of the same area against a post",
        "a small open area on the shoulder",
        "hair loss along one side",
        "swelling around a recent skin wound",
    ),
    "behaviour": (
        "withdrawal from the rest of the group",
        "failure to respond to a familiar call",
        "repeated head pressing against a surface",
        "unusual restlessness during handling",
        "a marked change in the usual routine",
    ),
    "unknown": (
        "an unexplained change from the usual condition",
        "an observation that cannot be localised",
        "an unspecified report of illness without further detail",
        "a change that the observer cannot describe",
        "an observation with no clear symptom recorded",
    ),
}

# Hold out symptom concepts, not just alternate sentence frames.
SYMPTOMS_BY_SPLIT: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {category: phrases[:3] for category, phrases in SYMPTOMS.items()},
    "validation": {category: (phrases[3],) for category, phrases in SYMPTOMS.items()},
    "test": {category: (phrases[4],) for category, phrases in SYMPTOMS.items()},
}

SYSTEMIC_STATES = ("normal", "mild", "critical")
FUNCTION_STATES = ("normal", "limited", "failed")
TRAJECTORY_STATES = ("improving", "stable", "worsening")

SYSTEMIC_TEXT = {
    "train": {
        "normal": (
            "temperature and alertness are within the usual range",
            "no general change is observed",
        ),
        "mild": (
            "there is a mild change in general condition",
            "the animal seems slightly dull",
        ),
        "critical": (
            "the animal is markedly dull and has a very high temperature",
            "there is a severe change in general condition",
        ),
    },
    "validation": {
        "normal": (
            "general observations remain unremarkable",
            "alertness and temperature appear usual",
        ),
        "mild": (
            "the overall condition is somewhat reduced",
            "the observer notes a small general change",
        ),
        "critical": (
            "the animal is difficult to rouse and feels extremely hot",
            "the general condition is severely compromised",
        ),
    },
    "test": {
        "normal": (
            "no broader abnormality was noticed",
            "the general check did not raise concern",
        ),
        "mild": (
            "the wider check suggests a modest change",
            "the animal is a little less responsive overall",
        ),
        "critical": (
            "the animal barely responds and is extremely hot",
            "the wider check indicates a major systemic problem",
        ),
    },
}

FUNCTION_TEXT = {
    "train": {
        "normal": (
            "it can carry out its usual activities",
            "normal function is maintained",
        ),
        "limited": (
            "some usual activities are now difficult",
            "its normal activity is reduced",
        ),
        "failed": (
            "it cannot perform a basic usual activity",
            "it is unable to carry out a normal function",
        ),
    },
    "validation": {
        "normal": (
            "usual activity remains possible",
            "it continues to function normally",
        ),
        "limited": (
            "it manages only part of its usual activity",
            "ordinary movement or activity is restricted",
        ),
        "failed": (
            "it cannot complete a basic activity",
            "a normal function has been lost",
        ),
    },
    "test": {
        "normal": (
            "there is no limitation in its usual activity",
            "it remains able to do what it normally does",
        ),
        "limited": (
            "its ordinary activity is noticeably restricted",
            "it struggles with part of its usual routine",
        ),
        "failed": (
            "it is unable to complete an essential activity",
            "it has lost the ability to perform a basic function",
        ),
    },
}

TRAJECTORY_TEXT = {
    "train": {
        "improving": (
            "the change has been easing",
            "the observation is less pronounced than before",
        ),
        "stable": (
            "the observation has remained much the same",
            "no clear change has been noted over time",
        ),
        "worsening": (
            "the change has become more pronounced",
            "the observation has continued to deteriorate",
        ),
    },
    "validation": {
        "improving": (
            "the signs have started to settle",
            "the situation appears to be getting better",
        ),
        "stable": ("the pattern has not shifted", "the same findings persist"),
        "worsening": (
            "the signs are becoming more apparent",
            "the situation has been getting worse",
        ),
    },
    "test": {
        "improving": (
            "there has been a gradual recovery",
            "the earlier change is now less evident",
        ),
        "stable": (
            "the findings are still at the previous level",
            "the situation has not materially changed",
        ),
        "worsening": (
            "the findings have intensified",
            "the situation is deteriorating further",
        ),
    },
}

RECORD_FRAMES = {
    split: ("Animal {animal} was observed with {symptom}. {systemic}. {function}. {trajectory}.",)
    for split in SYMPTOMS_BY_SPLIT
}

ANIMAL_IDS = tuple(f"A{i:03d}" for i in range(1, 241))
MAX_UNIQUE_NOTES_BY_SPLIT = {
    split: (
        len(ANIMAL_IDS)
        * sum(len(values) for values in concepts.values())
        * len(SYSTEMIC_STATES)
        * 2
        * len(FUNCTION_STATES)
        * 2
        * len(TRAJECTORY_STATES)
        * 2
    )
    for split, concepts in SYMPTOMS_BY_SPLIT.items()
}
MAX_UNIQUE_NOTES = max(MAX_UNIQUE_NOTES_BY_SPLIT.values())


def derive_severity(systemic: str, function: str, trajectory: str) -> str:
    """Apply the policy to structured attributes, not to a preselected label."""

    if systemic == "critical" or function == "failed":
        return "urgent"
    if systemic == "mild" or function == "limited" or trajectory == "worsening":
        return "watch"
    return "routine"


def build_note(
    rng: random.Random,
    category: str,
    systemic: str,
    function: str,
    trajectory: str,
    *,
    split: str = "train",
) -> str:
    if split not in RECORD_FRAMES:
        raise ValueError(f"unknown split family: {split!r}")
    return rng.choice(RECORD_FRAMES[split]).format(
        animal=rng.choice(ANIMAL_IDS),
        symptom=rng.choice(SYMPTOMS_BY_SPLIT[split][category]),
        systemic=rng.choice(SYSTEMIC_TEXT[split][systemic]),
        function=rng.choice(FUNCTION_TEXT[split][function]),
        trajectory=rng.choice(TRAJECTORY_TEXT[split][trajectory]),
    )


def labelled_note(rng: random.Random, *, split: str = "train") -> tuple[str, str, str]:
    """Sample independent scenario attributes and derive target labels."""

    category = rng.choices(CATEGORIES, weights=[18, 18, 18, 16, 16, 14], k=1)[0]
    systemic = rng.choices(SYSTEMIC_STATES, weights=[0.60, 0.25, 0.15], k=1)[0]
    function = rng.choices(FUNCTION_STATES, weights=[0.60, 0.25, 0.15], k=1)[0]
    trajectory = rng.choices(TRAJECTORY_STATES, weights=[0.25, 0.50, 0.25], k=1)[0]
    severity = derive_severity(systemic, function, trajectory)
    return (
        build_note(rng, category, systemic, function, trajectory, split=split),
        category,
        severity,
    )


def make_records(
    count: int,
    seed: int,
    *,
    seen_notes: set[str] | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Generate unique instruction records for one independent split family."""

    if count < 1:
        raise ValueError("record count must be at least 1")
    if split not in SYMPTOMS_BY_SPLIT:
        raise ValueError(f"split must be one of {tuple(SYMPTOMS_BY_SPLIT)}")
    capacity = MAX_UNIQUE_NOTES_BY_SPLIT[split]
    if count > capacity:
        raise ValueError(
            f"requested {count} records for {split}, but that split has only "
            f"{capacity} unique scenario combinations"
        )

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen = seen_notes if seen_notes is not None else set()
    attempts = 0
    max_attempts = max(10_000, count * 100)

    while len(records) < count and attempts < max_attempts:
        attempts += 1
        note, category, severity = labelled_note(rng, split=split)
        if note in seen:
            continue
        seen.add(note)
        records.append(
            {
                "instruction": INSTRUCTION,
                "input": note,
                "output": (f"sev={severity} cat={category} act={ACTION_BY_SEVERITY[severity]}"),
            }
        )

    if len(records) != count:
        raise RuntimeError(
            f"generated only {len(records)} of {count} unique records after "
            f"{attempts} attempts; reduce split sizes or expand the scenario space"
        )

    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/v3"))
    parser.add_argument("--train", type=int, default=480)
    parser.add_argument("--validation", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Each split gets distinct symptom concepts; exact-note disjointness is an extra guard.
    seen_notes: set[str] = set()
    train = make_records(args.train, args.seed, seen_notes=seen_notes, split="train")
    validation = make_records(
        args.validation, args.seed + 1, seen_notes=seen_notes, split="validation"
    )
    test = make_records(args.test, args.seed + 2, seen_notes=seen_notes, split="test")

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    write_jsonl(args.out_dir / "test.jsonl", test)
    manifest = {
        "dataset_version": "v3",
        "seed": args.seed,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "symptom_concepts_by_split": {
            split: {category: list(phrases) for category, phrases in categories.items()}
            for split, categories in SYMPTOMS_BY_SPLIT.items()
        },
        "severity_policy": {
            "urgent": "systemic=critical OR function=failed",
            "watch": "otherwise systemic=mild OR function=limited OR trajectory=worsening",
            "routine": "otherwise",
        },
        "action_by_severity": ACTION_BY_SEVERITY,
        "synthetic_data_warning": "Engineering fixture only; not veterinary guidance.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(train)} train + {len(validation)} validation + "
        f"{len(test)} test records to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
