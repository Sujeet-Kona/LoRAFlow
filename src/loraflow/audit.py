"""Dataset audit helpers: duplication, template leakage, and memorization baselines.

Torch-free on purpose, like :mod:`loraflow.data` and :mod:`loraflow.metrics`, so the
audit runs in the light CI job and on a machine with no ML extras installed.

The question this module answers is narrow and important: is a held-out split actually
novel, or does it re-use surface forms the model has already been optimised on? Two
numbers decide that:

``template_leakage``
    share of held-out notes whose id-masked skeleton also occurs in the training notes.
    At 1.0 the split contains no new *shape*, only new tokens.

``lexical_baseline``
    accuracy of a memorizing 1-nearest-neighbour lookup that never trains at all. Any
    fine-tuned score at or below it is evidence of pattern reuse, not learned judgement.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any

__all__ = [
    "skeleton",
    "note_tokens",
    "instruction_inventory",
    "class_distribution",
    "exact_duplicates",
    "template_inventory",
    "split_overlap",
    "near_duplicate_stats",
    "lexical_baseline",
    "audit_records",
]

_ID_TOKEN = re.compile(r"\b[A-Z]{1,3}#?\d+\b", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z][a-z0-9_\-']+")
# The tokens skeleton() substitutes for ids and numbers. They appear in every shape, so
# leaving them in the token sets would inflate every similarity by a constant.
_PLACEHOLDERS = frozenset({"id", "n"})


def skeleton(text: str) -> str:
    """Collapse a note to its surface shape: ids and numbers masked, case folded.

    Matching is case-insensitive and applied before folding, otherwise a record id
    survives into the shape and every note looks novel.
    """

    cleaned = _ID_TOKEN.sub("<id>", text)
    cleaned = _NUMBER.sub("<n>", cleaned)
    return _WHITESPACE.sub(" ", cleaned.lower()).strip()


def note_tokens(text: str) -> set[str]:
    """Content words of a note, used for lexical similarity."""

    return set(_WORD.findall(skeleton(text))) - _PLACEHOLDERS


def _parse_output_fields(output: str) -> dict[str, str]:
    return dict(pair.split("=", 1) for pair in output.split() if "=" in pair)


def instruction_inventory(
    records: Sequence[dict[str, Any]], *, instruction_field: str = "instruction"
) -> dict[str, Any]:
    """How many distinct instructions the corpus actually contains.

    One repeated instruction means the model never has to generalise across task
    phrasings; it only has to generalise across notes.
    """

    counts = Counter(str(record.get(instruction_field, "")) for record in records)
    total = len(records)
    return {
        "distinct_instructions": len(counts),
        "dominant_instruction_share": round(counts.most_common(1)[0][1] / total, 4)
        if total
        else 0.0,
        "instruction_lengths": sorted(len(text) for text in counts),
    }


def class_distribution(
    records: Sequence[dict[str, Any]], *, output_field: str = "output"
) -> dict[str, Any]:
    """Per-field and joint label counts over raw records."""

    per_field: dict[str, Counter] = defaultdict(Counter)
    joint: Counter = Counter()
    unparsed = 0
    for record in records:
        fields = _parse_output_fields(str(record.get(output_field, "")))
        if not fields:
            unparsed += 1
            continue
        for key, value in fields.items():
            per_field[key][value] += 1
        joint[record.get(output_field)] += 1

    total = len(records)
    return {
        "records": total,
        "unparsed_outputs": unparsed,
        "unique_output_strings": len(joint),
        "fields": {
            key: {
                value: {"count": count, "share": round(count / total, 4)}
                for value, count in sorted(counts.items(), key=lambda kv: -kv[1])
            }
            for key, counts in sorted(per_field.items())
        },
        "top_outputs": [
            {"output": output, "count": count} for output, count in joint.most_common(5)
        ],
    }


def exact_duplicates(
    records: Sequence[dict[str, Any]], *, input_field: str = "input"
) -> dict[str, Any]:
    """Whole-note and note+label duplication inside one split."""

    notes = [str(record.get(input_field, "")) for record in records]
    pairs = [
        f"{note}\t{record.get('output', '')}" for note, record in zip(notes, records, strict=True)
    ]
    note_counts = Counter(notes)
    repeated = {note: count for note, count in note_counts.items() if count > 1}
    return {
        "records": len(records),
        "unique_notes": len(note_counts),
        "repeated_notes": len(repeated),
        "most_repeated_note_count": max(note_counts.values()) if note_counts else 0,
        "unique_note_label_pairs": len(set(pairs)),
    }


def template_inventory(
    records: Sequence[dict[str, Any]], *, input_field: str = "input", output_field: str = "output"
) -> dict[str, Any]:
    """How many distinct surface shapes carry the labels, and whether shape predicts label.

    ``shape_label_conflicts`` counts skeletons paired with more than one label set; a
    corpus where every skeleton maps to one label can be solved by shape lookup alone.
    """

    shapes: dict[str, Counter] = defaultdict(Counter)
    shape_notes: dict[str, set] = defaultdict(set)
    for record in records:
        note = str(record.get(input_field, ""))
        shapes[skeleton(note)][str(record.get(output_field, ""))] += 1
        shape_notes[skeleton(note)].add(note)

    total = len(records)
    counts = sorted(Counter({shape: sum(c.values()) for shape, c in shapes.items()}).values())
    conflicting = sum(1 for c in shapes.values() if len(c) > 1)
    return {
        "records": total,
        "distinct_skeletons": len(shapes),
        "skeletons_per_record": round(len(shapes) / total, 4) if total else 0.0,
        "most_used_skeleton_share": round(counts[-1] / total, 4) if counts else 0.0,
        "median_skeleton_reuse": counts[len(counts) // 2] if counts else 0,
        "shape_label_conflicts": conflicting,
        "shapes_that_fix_the_label": len(shapes) - conflicting,
    }


def split_overlap(
    train: Sequence[dict[str, Any]],
    held: Sequence[dict[str, Any]],
    *,
    input_field: str = "input",
) -> dict[str, Any]:
    """Exact, shape-level and token-level overlap between two splits."""

    train_notes = {str(record.get(input_field, "")) for record in train}
    held_notes = [str(record.get(input_field, "")) for record in held]
    train_shapes = {skeleton(note) for note in train_notes}
    held_shapes = [skeleton(note) for note in held_notes]
    train_words = {word for note in train_notes for word in note_tokens(note)}
    held_words = {word for note in held_notes for word in note_tokens(note)}

    shared = sum(1 for shape in held_shapes if shape in train_shapes)
    novel = sum(1 for shape in held_shapes if shape not in train_shapes)
    return {
        "held_records": len(held_notes),
        "exact_shared_notes": len(train_notes & set(held_notes)),
        "shared_skeletons": len(train_shapes & set(held_shapes)),
        "held_skeletons_already_in_train": shared,
        "template_leakage": round(shared / len(held_shapes), 4) if held_shapes else 0.0,
        "held_skeletons_novel": novel,
        "novel_words_in_held": len(held_words - train_words),
        "held_word_coverage_by_train": round(len(held_words & train_words) / len(held_words), 4)
        if held_words
        else 0.0,
    }


def near_duplicate_stats(
    train: Sequence[dict[str, Any]],
    held: Sequence[dict[str, Any]],
    *,
    input_field: str = "input",
    sample: int = 60,
) -> dict[str, Any]:
    """Best word-level Jaccard and character ratio from each held note to any train note.

    ``sample`` bounds the quadratic cost; the pool searched is always the full train split.
    """

    train_token_sets = [note_tokens(str(record.get(input_field, ""))) for record in train]
    train_notes = [str(record.get(input_field, "")) for record in train]
    held_notes = [str(record.get(input_field, "")) for record in held]
    if sample and sample < len(held_notes):
        # Deterministic spread over the split rather than only its head.
        step = len(held_notes) / sample
        held_notes = [held_notes[int(i * step)] for i in range(sample)]

    jaccards: list[float] = []
    ratios: list[float] = []
    worst: list[dict[str, Any]] = []
    for note in held_notes:
        tokens = note_tokens(note)
        best_jaccard, best_index = 0.0, -1
        for index, other in enumerate(train_token_sets):
            union = tokens | other
            score = len(tokens & other) / len(union) if union else 0.0
            if score > best_jaccard:
                best_jaccard, best_index = score, index
        char_ratio = (
            SequenceMatcher(None, skeleton(note), skeleton(train_notes[best_index])).ratio()
            if best_index >= 0
            else 0.0
        )
        jaccards.append(best_jaccard)
        ratios.append(char_ratio)
        worst.append(
            {"held": note, "jaccard": round(best_jaccard, 4), "ratio": round(char_ratio, 4)}
        )

    ordered = sorted(worst, key=lambda item: -item["jaccard"])
    count = len(jaccards)
    return {
        "sampled": count,
        "mean_best_jaccard": round(sum(jaccards) / count, 4) if count else 0.0,
        "held_within_0_9_of_a_train_note": sum(1 for score in jaccards if score >= 0.9),
        "held_within_0_9_share": round(sum(1 for score in jaccards if score >= 0.9) / count, 4)
        if count
        else 0.0,
        "mean_char_ratio_at_best_match": round(sum(ratios) / count, 4) if count else 0.0,
        "closest_three": ordered[:3],
    }


def lexical_baseline(
    train: Sequence[dict[str, Any]],
    held: Sequence[dict[str, Any]],
    *,
    input_field: str = "input",
    output_field: str = "output",
) -> dict[str, Any]:
    """Accuracy of a pure memorization strategy on the held-out notes.

    1-nearest-neighbour by word overlap, copying the neighbour's whole label line, plus
    the majority label line. A fine-tuned model that does not beat the neighbour number
    has learned the lookup table, not the task.
    """

    train_tokens = [note_tokens(str(record.get(input_field, ""))) for record in train]
    train_outputs = [str(record.get(output_field, "")) for record in train]
    majority = Counter(train_outputs).most_common(1)[0][0] if train_outputs else ""

    field_totals: dict[str, int] = defaultdict(int)
    field_hits: dict[str, int] = defaultdict(int)
    field_majority_hits: dict[str, int] = defaultdict(int)
    train_field_counts: dict[str, Counter] = defaultdict(Counter)
    for output in train_outputs:
        for key, value in _parse_output_fields(output).items():
            train_field_counts[key][value] += 1
    field_majority = {
        key: counts.most_common(1)[0][0]
        for key, counts in train_field_counts.items()
        if counts
    }
    neighbour_exact = 0
    majority_exact = 0
    held_count = 0

    for record in held:
        gold = str(record.get(output_field, ""))
        fields = _parse_output_fields(gold)
        if not fields:
            continue
        held_count += 1

        tokens = note_tokens(str(record.get(input_field, "")))
        best_score, best_output = -1.0, ""
        for index, other in enumerate(train_tokens):
            # Strict > keeps the earliest training record as the tie-break.
            union = tokens | other
            score = len(tokens & other) / len(union) if union else 0.0
            if score > best_score:
                best_score, best_output = score, train_outputs[index]

        prediction = _parse_output_fields(best_output)
        if best_output == gold:
            neighbour_exact += 1
        if majority == gold:
            majority_exact += 1
        for key, value in fields.items():
            field_totals[key] += 1
            if prediction.get(key) == value:
                field_hits[key] += 1
            if field_majority.get(key) == value:
                field_majority_hits[key] += 1

    return {
        "held_records": held_count,
        "exact_match_1nn": round(neighbour_exact / held_count, 4) if held_count else 0.0,
        "exact_match_majority": round(majority_exact / held_count, 4) if held_count else 0.0,
        "per_field_1nn": {
            key: round(field_hits[key] / field_totals[key], 4)
            for key in sorted(field_totals)
            if field_totals[key]
        },
        "per_field_majority": {
            key: round(field_majority_hits[key] / field_totals[key], 4)
            for key in sorted(field_totals)
            if field_totals[key]
        },
        "majority_field_values": field_majority,
        "majority_output": majority,
    }


def audit_records(
    train: Sequence[dict[str, Any]],
    held_splits: dict[str, Sequence[dict[str, Any]]],
    *,
    input_field: str = "input",
    output_field: str = "output",
) -> dict[str, Any]:
    """Full audit of one training split against any number of held-out splits."""

    return {
        "train": {
            "size": len(train),
            "instructions": instruction_inventory(train),
            "duplicates": exact_duplicates(train, input_field=input_field),
            "labels": class_distribution(train, output_field=output_field),
            "templates": template_inventory(
                train, input_field=input_field, output_field=output_field
            ),
        },
        "held": {
            name: {
                "size": len(records),
                "instructions": instruction_inventory(records),
                "duplicates": exact_duplicates(records, input_field=input_field),
                "labels": class_distribution(records, output_field=output_field),
                "templates": template_inventory(
                    records, input_field=input_field, output_field=output_field
                ),
                "overlap_with_train": split_overlap(train, records, input_field=input_field),
                "near_duplicates_vs_train": near_duplicate_stats(
                    train, records, input_field=input_field
                ),
                "memorization_baseline": lexical_baseline(
                    train, records, input_field=input_field, output_field=output_field
                ),
            }
            for name, records in held_splits.items()
        },
    }


def read_records(path: Any) -> list[dict[str, Any]]:
    """Thin wrapper so the audit works on raw records, not formatted prompts."""

    from loraflow.data import load_jsonl

    records: Iterable[dict[str, Any]] = load_jsonl(path, text_field="input")
    return list(records)
