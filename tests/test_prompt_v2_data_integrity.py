import hashlib
import json

import pytest

from loraflow.data_integrity import SPLITS, verify_track


def _write_track(directory, *, record=None):
    directory.mkdir(parents=True)
    row = record or {
        "instruction": "Classify the note",
        "input": "Example input",
        "output": "severity=watch",
    }
    counts = {}
    hashes = {}

    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        raw = (json.dumps(row) + "\n").encode("utf-8")
        path.write_bytes(raw)
        counts[split] = 1
        hashes[split] = hashlib.sha256(raw).hexdigest()

    (directory / "manifest.json").write_text(
        json.dumps({"track": "A", "counts": counts, "sha256": hashes}),
        encoding="utf-8",
    )


def test_verify_track_accepts_valid_manifest_and_splits(tmp_path):
    _write_track(tmp_path / "track_a")

    assert verify_track(tmp_path, "a") == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_verify_track_rejects_modified_split_hash(tmp_path):
    directory = tmp_path / "track_a"
    _write_track(directory)
    (directory / "test.jsonl").write_text(
        '{"instruction":"tampered","input":"x","output":"y"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_track(tmp_path, "a")


def test_verify_track_rejects_invalid_record_even_with_matching_hash(tmp_path):
    directory = tmp_path / "track_a"
    _write_track(directory)
    invalid = b'{"instruction":"Classify","input":"note"}\n'
    path = directory / "validation.jsonl"
    path.write_bytes(invalid)

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["validation"] = hashlib.sha256(invalid).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected object with"):
        verify_track(tmp_path, "a")
