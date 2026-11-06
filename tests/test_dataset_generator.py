"""Tests for the deterministic synthetic corpus generator."""

import runpy
from pathlib import Path

import pytest

_GENERATOR = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "make_dataset.py")
)
make_records = _GENERATOR["make_records"]
derive_severity = _GENERATOR["derive_severity"]


def test_generated_partitions_are_exact_note_disjoint() -> None:
    seen: set[str] = set()
    train = make_records(40, seed=7, seen_notes=seen, split="train")
    validation = make_records(20, seed=8, seen_notes=seen, split="validation")
    test = make_records(20, seed=9, seen_notes=seen, split="test")

    notes = [{record["input"] for record in split} for split in (train, validation, test)]
    assert notes[0].isdisjoint(notes[1])
    assert notes[0].isdisjoint(notes[2])
    assert notes[1].isdisjoint(notes[2])


def test_generator_is_deterministic_without_shared_state() -> None:
    assert make_records(20, seed=11) == make_records(20, seed=11)


@pytest.mark.parametrize(
    ("systemic", "function", "trajectory", "expected"),
    [
        ("critical", "normal", "improving", "urgent"),
        ("normal", "failed", "stable", "urgent"),
        ("mild", "normal", "improving", "watch"),
        ("normal", "limited", "stable", "watch"),
        ("normal", "normal", "worsening", "watch"),
        ("normal", "normal", "stable", "routine"),
        ("normal", "normal", "improving", "routine"),
    ],
)
def test_severity_is_derived_from_scenario_attributes(
    systemic: str, function: str, trajectory: str, expected: str
) -> None:
    assert derive_severity(systemic, function, trajectory) == expected


def test_split_families_use_distinct_surface_frames() -> None:
    train = make_records(30, seed=21, split="train")
    validation = make_records(30, seed=22, split="validation")
    test = make_records(30, seed=23, split="test")

    # The opening record frame is intentionally split-specific.
    assert all(record["input"].startswith("Animal ") for record in train)
    assert all(record["input"].startswith("Observation for animal ") for record in validation)
    assert all(record["input"].startswith("Case ") for record in test)


@pytest.mark.parametrize("count", [0, -1])
def test_generator_rejects_non_positive_counts(count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_records(count, seed=1)


def test_generator_rejects_counts_above_unique_capacity() -> None:
    with pytest.raises(ValueError, match="unique scenario combinations"):
        make_records(200000, seed=1)


def test_generator_rejects_unknown_split_family() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        make_records(1, seed=1, split="unknown")
