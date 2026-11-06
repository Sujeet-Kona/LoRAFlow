from __future__ import annotations

from collections import Counter

from loraflow.v4_benchmarks import (
    ACTIONS,
    CATEGORIES,
    INSTRUCTION,
    SEVERITIES,
    audit,
    generate,
    target_severity_counts,
)


def _make_data(track: str, sizes: tuple[int, int, int], seed: int):
    shared_concept_count = min(48, min(sizes)) if track == "B" else None
    return {
        split: generate(track, split, count, seed + index, shared_concept_count)
        for index, (split, count) in enumerate(
            zip(("train", "validation", "test"), sizes, strict=True)
        )
    }


def _assert_stratified(data):
    for rows in data.values():
        observed = Counter(row["output"].split()[0].split("=")[1] for row in rows)
        assert dict(observed) == target_severity_counts(len(rows))
        assert all(
            ACTIONS[row["output"].split()[0].split("=")[1]]
            == row["output"].split()[2].split("=")[1]
            for row in rows
        )


def test_v4_track_a_holds_out_concepts_but_shares_severity_wording() -> None:
    data = _make_data("A", (60, 24, 24), 11)
    report = audit("A", data)
    train = set(report["concepts"]["train"])
    validation = set(report["concepts"]["validation"])
    test = set(report["concepts"]["test"])
    assert len(train) == 30
    assert len(validation) == 6
    assert len(test) == 12
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert report["severity_cue_sets"]["train"] == report["severity_cue_sets"]["validation"]
    _assert_stratified(data)


def test_v4_track_b_shares_concepts_but_holds_out_severity_wording() -> None:
    data = _make_data("B", (60, 24, 24), 21)
    report = audit("B", data)
    train_concepts = set(report["concepts"]["train"])
    validation_concepts = set(report["concepts"]["validation"])
    test_concepts = set(report["concepts"]["test"])
    assert len(train_concepts) == 24
    assert train_concepts == validation_concepts == test_concepts
    assert not set(report["severity_cue_sets"]["train"]) & set(
        report["severity_cue_sets"]["validation"]
    )
    assert not set(report["severity_cue_sets"]["train"]) & set(report["severity_cue_sets"]["test"])
    _assert_stratified(data)


def test_v4_track_c_combines_both_distribution_shifts() -> None:
    data = _make_data("C", (60, 24, 24), 31)
    report = audit("C", data)
    assert not set(report["concepts"]["train"]) & set(report["concepts"]["validation"])
    assert not set(report["concepts"]["train"]) & set(report["concepts"]["test"])
    assert not set(report["severity_cue_sets"]["train"]) & set(
        report["severity_cue_sets"]["validation"]
    )
    _assert_stratified(data)


def test_default_split_sizes_have_matching_severity_proportions() -> None:
    for track, seed in (("A", 41), ("B", 141), ("C", 241)):
        data = _make_data(track, (600, 120, 150), seed)
        report = audit(track, data)
        _assert_stratified(data)
        assert report["policy_checks"] == "passed"
        if track == "B":
            assert all(len(report["concepts"][split]) == 48 for split in data)


def test_generated_inputs_are_unique_within_and_across_splits() -> None:
    data = _make_data("B", (60, 24, 24), 51)
    notes = [row["input"] for rows in data.values() for row in rows]
    assert len(notes) == len(set(notes))


def test_severity_targets_sum_to_requested_count() -> None:
    for count in (1, 24, 60, 120, 150, 600):
        assert sum(target_severity_counts(count).values()) == count
        assert set(target_severity_counts(count)) == set(SEVERITIES)


def test_v4_instruction_enumerates_the_complete_output_schema() -> None:
    allowed_labels = (*SEVERITIES, *CATEGORIES, *ACTIONS.values())
    assert all(label in INSTRUCTION for label in allowed_labels)
    assert "exactly one line" in INSTRUCTION
    assert "Do not add explanations" in INSTRUCTION
