"""Verify committed Prompt-v2 datasets against their manifests."""

from __future__ import annotations

from pathlib import Path

from loraflow.data_integrity import TRACKS, verify_track


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "v4-prompt-v2"


def main() -> None:
    for track in TRACKS:
        counts = verify_track(DATA_ROOT, track)
        summary = ", ".join(f"{split}={count}" for split, count in counts.items())
        print(f"Prompt-v2 track {track.upper()}: verified ({summary})")


if __name__ == "__main__":
    main()
