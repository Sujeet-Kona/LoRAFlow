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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loraflow.audit import audit_records, read_records  # noqa: E402


def _parse_pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--held expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


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
    args = parser.parse_args()

    train = read_records(args.train)
    paths = dict(args.held)
    held = {name: read_records(path) for name, path in paths.items()}
    audit = audit_records(train, held)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "files": {"train": str(args.train), **{k: str(v) for k, v in paths.items()}},
                "summary": _summary(audit),
                "full_audit": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
