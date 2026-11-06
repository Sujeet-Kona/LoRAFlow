"""Integrity checks for the committed Prompt-v2 benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")
TRACKS = ("a", "b", "c")
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "v4-prompt-v2"


def verify_track(data_root: Path, track: str) -> dict[str, int]:
    """Verify split hashes, row counts, and basic JSONL record structure."""
    directory = data_root / f"track_{track}"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("track") != track.upper():
        raise ValueError(
            f"{manifest_path}: expected track {track.upper()!r}, found {manifest.get('track')!r}"
        )

    counts = manifest.get("counts")
    hashes = manifest.get("sha256")
    if not isinstance(counts, dict) or not isinstance(hashes, dict):
        raise ValueError(f"{manifest_path}: missing counts or sha256 mapping")

    verified: dict[str, int] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split: {path}")

        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        expected_hash = hashes.get(split)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{path}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
            )

        rows = [line for line in raw.splitlines() if line.strip()]
        expected_count = counts.get(split)
        if len(rows) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} records, got {len(rows)}")

        for line_number, line in enumerate(rows, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            required = {"instruction", "input", "output"}
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(
                    f"{path}:{line_number}: expected object with {sorted(required)} fields"
                )
            if not all(isinstance(record[key], str) and record[key].strip() for key in required):
                raise ValueError(
                    f"{path}:{line_number}: instruction, input, and output "
                    "must be non-empty strings"
                )

        verified[split] = len(rows)

    return verified


def main() -> None:
    for track in TRACKS:
        counts = verify_track(DATA_ROOT, track)
        summary = ", ".join(f"{split}={count}" for split, count in counts.items())
        print(f"Prompt-v2 track {track.upper()}: verified ({summary})")
