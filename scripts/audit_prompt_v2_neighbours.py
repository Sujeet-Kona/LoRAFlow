"""Run an exhaustive nearest-neighbour audit for the committed Prompt-v2 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loraflow.audit import near_duplicate_stats, read_records  # noqa: E402

TRACKS = ("a", "b", "c")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "v4-prompt-v2",
        help="Directory containing track_a, track_b, and track_c",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "ci-smoke" / "prompt-v2-nearest-neighbours.json",
    )
    args = parser.parse_args()

    results = {
        track.upper(): near_duplicate_stats(
            read_records(args.data_root / f"track_{track}" / "train.jsonl"),
            read_records(args.data_root / f"track_{track}" / "test.jsonl"),
            sample=0,  # zero means every held-out record, not the default 60-note sample
        )
        for track in TRACKS
    }
    payload = {
        "scope": "all 150 held-out records compared against all 600 training records per track",
        "similarity": "word-level Jaccard over loraflow.audit.note_tokens",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    for track, result in results.items():
        print(
            f"Track {track}: {result['sampled']} test records; "
            f"Jaccard >= 0.9: {result['held_within_0_9_of_a_train_note']}; "
            f"mean best Jaccard: {result['mean_best_jaccard']:.4f}"
        )
    print(f"Full audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
