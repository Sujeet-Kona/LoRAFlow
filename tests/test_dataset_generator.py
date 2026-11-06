"""Tests for the deterministic synthetic corpus generator."""

import pytest

from scripts.make_dataset import make_records


def test_generated_partitions_are_exact_note_disjoint() -> None:
    seen: set[str] = set()
    train = make_records(40, seed=7, seen_notes=seen)
    validation = make_records(20, seed=8, seen_notes=seen)
    test = make_records(20, seed=9, seen_notes=seen)

    notes = [{record["input"] for record in split} for split in (train, validation, test)]
    assert notes[0].isdisjoint(notes[1])
    assert notes[0].isdisjoint(notes[2])
    assert notes[1].isdisjoint(notes[2])


def test_generator_is_deterministic_without_shared_state() -> None:
    assert make_records(20, seed=11) == make_records(20, seed=11)


@pytest.mark.parametrize("count", [0, -1])
def test_generator_rejects_non_positive_counts(count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_records(count, seed=1)


def test_generator_rejects_counts_above_unique_capacity() -> None:
    with pytest.raises(ValueError, match="unique note combinations"):
        make_records(122401, seed=1)
