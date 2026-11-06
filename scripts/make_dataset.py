"""Generate the bundled demo corpus for LoRAFlow.

The corpus is a small, deterministic "animal-health note triage" task: each
record asks the model to reduce a free-text barn note to a fixed one-line
schema. The schema tokens (``sev=``/``cat=``/``act=`` and the action codes)
are intentionally non-standard, so a base model cannot guess them from
pretraining alone -- which makes the baseline-versus-fine-tuned delta real and
measurable rather than assumed.

Everything comes from a fixed seed, so rerunning the script reproduces the
exact same files.

Usage:
    python scripts/make_dataset.py --out-dir data --train 480 --validation 120
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "Read the barn health note and label it with severity, category and action. "
    "Reply with exactly one line in the form sev=<severity> cat=<category> "
    "act=<action>. Use only the codes defined by the triage table."
)

CATEGORIES = ("mobility", "feeding", "respiratory", "skin", "behaviour", "unknown")
SEVERITIES = ("routine", "watch", "urgent")

ACTION_BY_SEVERITY = {
    "routine": "log_only",
    "watch": "recheck_12h",
    "urgent": "flag_vet_now",
}

# Category cues: phrase pool per category. The category label is decided by the
# dominant cue present in the note.
CATEGORY_CUES: dict[str, list[list[str]]] = {
    "mobility": [
        ["is limping on the", "front left leg"],
        ["will not put weight on the", "back right hoof"],
        ["stiff when standing up from the", "resting stall"],
        ["knocks the", "left hind leg", "against the feeder while walking"],
        ["has a sore", "front right fetlock"],
    ],
    "feeding": [
        ["left the", "morning ration untouched"],
        ["is eating much less", "than the rest of the herd"],
        ["spit out the", "fresh silage"],
        ["drank only a little", "water since dawn"],
        ["lost condition at the", "ribs and spine"],
    ],
    "respiratory": [
        ["is coughing", "through the night"],
        ["has a runny", "nose and watery eyes"],
        ["breathes fast even while", "standing still"],
        ["makes a whistling", "sound when breathing out"],
        ["has a dry", "hack at morning roll call"],
    ],
    "skin": [
        ["has a raw", "patch behind the ear"],
        ["is scratching the", "tail head until it is red"],
        ["shows a crusty", "lesion on the belly"],
        ["has flies landing on the", "wound at the shoulder"],
        ["lost hair along the", "left flank"],
    ],
    "behaviour": [
        ["stands apart from the", "group at the water trough"],
        ["is head-pressing against the", "wall of the pen"],
        ["will not follow the", "herd out to pasture"],
        ["is grinding teeth and", "refusing to move"],
        ["shows unusual", "aggression at feeding time"],
    ],
}

SEVERITY_CUES: dict[str, list[str]] = {
    "urgent": [
        "The animal cannot stand up.",
        "It collapsed in the stall.",
        "It is shaking with fever.",
        "It is breathing through its mouth at rest.",
        "It has not eaten for two days.",
        "The wound is still bleeding.",
        "Its temperature is running very high.",
    ],
    "watch": [
        "The sign is mild but has persisted for several days.",
        "It started yesterday morning and is slowly worsening.",
        "The change is small but it needs a closer look.",
        "It has been less lively than usual this week.",
        "Nothing severe so far, but it is worth rechecking.",
    ],
    "routine": [
        "Otherwise bright and alert.",
        "Moving normally, with no other change today.",
        "Comfortable at rest and drinking well.",
        "Eating fine apart from this note.",
        "No other change since the last check.",
    ],
}

# Notes with no locatable sign are the "unknown" category on purpose: the model
# has to learn to abstain instead of guessing a body system.
UNKNOWN_SUBJECTS = [
    "the daily check found no clear localising sign",
    "the herder reported something odd but could not describe it",
    "nothing specific stood out during the walk-through",
    "the only note is that the animal seems off today",
    "no leg, feed, breathing, skin or behaviour detail was recorded",
]

ANIMAL_IDS = [f"A{i:03d}" for i in range(1, 241)]


def build_note(rng: random.Random, category: str, severity: str) -> str:
    animal = rng.choice(ANIMAL_IDS)
    if category == "unknown":
        subject = rng.choice(UNKNOWN_SUBJECTS)
    else:
        subject = " ".join(rng.choice(CATEGORY_CUES[category]))
    lead = f"Animal {animal}: {subject}."
    return f"{lead} {rng.choice(SEVERITY_CUES[severity])}"


def labelled_note(rng: random.Random) -> tuple[str, str, str]:
    """Draw one (note, category, severity) pair with a realistic class mix."""
    category = rng.choices(CATEGORIES, weights=[20, 18, 18, 16, 16, 12], k=1)[0]
    severity = rng.choices(SEVERITIES, weights=[40, 40, 20], k=1)[0]
    return build_note(rng, category, severity), category, severity


def make_records(
    count: int, seed: int, *, seen_notes: set[str] | None = None
) -> list[dict[str, Any]]:
    """Generate unique records, optionally sharing a uniqueness set across splits."""
    maximum_unique_notes = len(ANIMAL_IDS) * sum(
        len(CATEGORY_CUES.get(category, UNKNOWN_SUBJECTS)) for category in CATEGORIES
    ) * sum(len(cues) for cues in SEVERITY_CUES.values())
    if count < 1:
        raise ValueError("record count must be at least 1")
    if count > maximum_unique_notes:
        raise ValueError(
            f"requested {count} records, but the generator has only "
            f"{maximum_unique_notes} unique note combinations"
        )

    rng = random.Random(seed)
    records = []
    seen = seen_notes if seen_notes is not None else set()
    while len(records) < count:
        note, category, severity = labelled_note(rng)
        if note in seen:
            continue
        seen.add(note)
        records.append(
            {
                "instruction": INSTRUCTION,
                "input": note,
                "output": f"sev={severity} cat={category} act={ACTION_BY_SEVERITY[severity]}",
            }
        )
    # Shuffle so a head/tail truncation still sees every class.
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=480)
    parser.add_argument("--validation", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Independent random seeds improve variation; a shared set guarantees that
    # no exact note can leak between train, validation, and test.
    seen_notes: set[str] = set()
    train = make_records(args.train, seed=args.seed, seen_notes=seen_notes)
    validation = make_records(args.validation, seed=args.seed + 1, seen_notes=seen_notes)
    test = make_records(args.test, seed=args.seed + 2, seen_notes=seen_notes)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    write_jsonl(args.out_dir / "test.jsonl", test)
    print(
        f"wrote {len(train)} train + {len(validation)} validation + "
        f"{len(test)} test records to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
