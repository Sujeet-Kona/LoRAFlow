"""Tests for the dataset audit helpers (stdlib only, no torch needed)."""

from loraflow.audit import (
    audit_records,
    class_distribution,
    exact_duplicates,
    instruction_inventory,
    lexical_baseline,
    near_duplicate_stats,
    note_tokens,
    skeleton,
    split_overlap,
    template_inventory,
)


def record(note: str, output: str) -> dict:
    return {"instruction": "Label the note.", "input": note, "output": output}


A1 = record(
    "Animal A222: stiff when standing up. It has been less lively this week.",
    "sev=watch cat=mobility act=recheck_12h",
)
A2 = record(
    "Animal A007: stiff when standing up. It has been less lively this week.",
    "sev=watch cat=mobility act=recheck_12h",
)
B1 = record(
    "Animal A007: left the morning ration untouched. Otherwise bright.",
    "sev=routine cat=feeding act=log_only",
)


def test_skeleton_masks_ids_regardless_of_case() -> None:
    assert skeleton("Animal A222: coughing") == skeleton("animal a001: coughing")
    assert "<id>" in skeleton("Animal A222: coughing")


def test_skeleton_masks_numbers_and_folds_whitespace() -> None:
    assert skeleton("row 12  of  pen 3") == skeleton("row 4 of pen 9")


def test_two_notes_differing_only_by_id_share_a_skeleton() -> None:
    assert skeleton(A1["input"]) == skeleton(A2["input"])
    assert skeleton(A1["input"]) != skeleton(B1["input"])


def test_note_tokens_drops_skeleton_placeholders() -> None:
    tokens = note_tokens(A1["input"])
    assert "id" not in tokens and "n" not in tokens
    assert "stiff" in tokens


def test_template_inventory_reports_shared_shapes_and_conflicts() -> None:
    records = [
        A1,
        A2,
        record(A2["input"], "sev=urgent cat=mobility act=flag_vet_now"),
        B1,
    ]

    inventory = template_inventory(records)

    # A1/A2/third share one shape (they differ only by id or label); B1 is another shape.
    assert inventory["distinct_skeletons"] == 2
    assert inventory["shape_label_conflicts"] == 1
    assert inventory["shapes_that_fix_the_label"] == 1


def test_split_overlap_flags_template_leakage() -> None:
    overlap = split_overlap([A1], [A2])

    assert overlap["exact_shared_notes"] == 0
    assert overlap["held_skeletons_already_in_train"] == 1
    assert overlap["template_leakage"] == 1.0
    assert overlap["held_skeletons_novel"] == 0


def test_split_overlap_reports_novel_shapes() -> None:
    overlap = split_overlap([A1], [B1])

    assert overlap["template_leakage"] == 0.0
    assert overlap["held_skeletons_novel"] == 1


def test_near_duplicate_stats_counts_lookalikes() -> None:
    stats = near_duplicate_stats([A1], [A2, B1], sample=0)

    assert stats["sampled"] == 2
    assert stats["held_within_0_9_of_a_train_note"] == 1
    assert stats["closest_three"][0]["jaccard"] > 0.9


def test_lexical_baseline_memorises_identical_notes() -> None:
    held = [
        A2,
        record(
            "Animal A900: nothing similar at all here.", "sev=urgent cat=unknown act=flag_vet_now"
        ),
    ]

    baseline = lexical_baseline([A1, B1], held)

    assert baseline["held_records"] == 2
    # The first held note differs from a training note only by id, so lookup copies it.
    assert baseline["exact_match_1nn"] == 0.5
    assert baseline["per_field_1nn"]["cat"] == 0.5
    assert baseline["majority_output"].startswith("sev=")


def test_lexical_baseline_is_deterministic() -> None:
    train = [
        A1,
        B1,
        record(
            "Animal A010: coughing through the night. It is shaking.",
            "sev=urgent cat=respiratory act=flag_vet_now",
        ),
    ]
    held = [A2]

    assert lexical_baseline(train, held) == lexical_baseline(train, held)


def test_class_distribution_shares_and_joint_counts() -> None:
    distribution = class_distribution([A1, A2, B1])

    assert distribution["records"] == 3
    assert distribution["unique_output_strings"] == 2
    assert distribution["fields"]["sev"]["watch"]["count"] == 2
    assert distribution["fields"]["sev"]["watch"]["share"] == 0.6667
    assert distribution["unparsed_outputs"] == 0


def test_exact_duplicates_and_instructions() -> None:
    duplicates = exact_duplicates([A1, A2, A1])
    assert duplicates["unique_notes"] == 2
    assert duplicates["repeated_notes"] == 1
    assert duplicates["most_repeated_note_count"] == 2

    inventory = instruction_inventory([A1, A2, B1])
    assert inventory["distinct_instructions"] == 1
    assert inventory["dominant_instruction_share"] == 1.0


def test_audit_records_covers_every_held_split() -> None:
    audit = audit_records([A1, B1], {"validation": [A2], "test": [B1]})

    assert set(audit["held"]) == {"validation", "test"}
    assert audit["train"]["size"] == 2
    assert audit["held"]["validation"]["overlap_with_train"]["template_leakage"] == 1.0
    assert audit["held"]["test"]["overlap_with_train"]["exact_shared_notes"] == 1
    assert "memorization_baseline" in audit["held"]["test"]
