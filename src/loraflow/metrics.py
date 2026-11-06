"""Scoring helpers for held-out evaluation.

Pure Python on purpose: every metric here is unit-testable without torch, a
tokenizer, or a model download, and the numbers reported by
:mod:`loraflow.evaluation` come straight from these functions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s.!?:;\"'`~]+$")
_MARKDOWN_FENCE = re.compile(r"^```[a-zA-Z0-9]*\n?|\n?```$")


def normalize_answer(text: str) -> str:
    """Canonical form used for exact-match comparison.

    Lower-cases, drops code fences, strips the ``### Response:`` scaffolding a
    base model often continues, collapses whitespace, and removes trailing
    punctuation. Two answers that differ only in that scaffolding compare equal.
    """

    cleaned = text.strip()
    # Split off the prompt scaffolding first: a base model that re-emits the
    # header leaves the code fence mid-string, where a ^ anchored pattern would
    # never find it.
    for marker in ("### Response:", "### response:"):
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, 1)[-1].strip()
    cleaned = _MARKDOWN_FENCE.sub("", cleaned).strip()
    cleaned = cleaned.lower()
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return _TRAILING_PUNCT.sub("", cleaned).strip()


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def prefix_match(prediction: str, gold: str, tokens: int = 3) -> bool:
    """True when the answer starts the same way, over at most ``tokens`` words.

    A cheap "did it head in the right direction" signal that stays informative
    even when the model rambles past the expected answer. A gold shorter than
    ``tokens`` only needs its own words to line up.
    """

    gold_words = normalize_answer(gold).split()
    if not gold_words:
        return False
    wanted = gold_words[:tokens]
    pred_words = normalize_answer(prediction).split()
    return pred_words[: len(wanted)] == wanted


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


@dataclass(frozen=True)
class GenerationScores:
    """Summary of scored free-form predictions."""

    examples: int
    exact_match: float
    prefix_match: float
    mean_prediction_chars: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "exact_match": round(self.exact_match, 4),
            "prefix_match": round(self.prefix_match, 4),
            "mean_prediction_chars": round(self.mean_prediction_chars, 1),
        }


def score_generations(
    predictions: Sequence[str], golds: Sequence[str], *, prefix_tokens: int = 3
) -> GenerationScores:
    if len(predictions) != len(golds):
        raise ValueError(f"predictions/golds length mismatch: {len(predictions)} vs {len(golds)}")
    pairs = zip(predictions, golds, strict=True)
    hits = sum(exact_match(p, g) for p, g in pairs)
    pairs = zip(predictions, golds, strict=True)
    prefixes = sum(prefix_match(p, g, prefix_tokens) for p, g in pairs)
    total = len(golds)
    lengths = sum(len(p) for p in predictions)
    return GenerationScores(
        examples=total,
        exact_match=_rate(hits, total),
        prefix_match=_rate(prefixes, total),
        mean_prediction_chars=_rate(lengths, total),
    )


def perplexity_from_loss(mean_loss: float) -> float:
    """Exponentiate a mean cross-entropy; overflow to inf instead of raising."""

    if mean_loss < 0:
        raise ValueError("mean loss must be non-negative")
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return math.inf


def token_accuracy(
    logits_argmax: Iterable[int], labels: Iterable[int], *, ignore_index: int = -100
) -> tuple[int, int]:
    """Compare forced-decoding predictions to labels; returns (correct, total).

    Positions masked with ``ignore_index`` are excluded, which is what makes the
    metric measure the answer span rather than the prompt.
    """

    correct = 0
    total = 0
    for index, (predicted, label) in enumerate(zip(logits_argmax, labels, strict=True)):
        if label == ignore_index:
            continue
        if index == 0:
            # Position 0 has no preceding token, so it is never a prediction.
            continue
        total += 1
        if predicted == label:
            correct += 1
    return correct, total


_FIELD_PAIR = re.compile(r"([a-z][a-z0-9_]*)=([^\s]+)")


def parse_fields(text: str) -> dict[str, str]:
    """Extract ``name=value`` pairs from a normalised answer.

    This is how per-field accuracy stays generic: it scores whatever key/value
    tokens the target answer uses rather than a hard-coded schema.
    """

    return {
        match.group(1): match.group(2) for match in _FIELD_PAIR.finditer(normalize_answer(text))
    }


def field_scores(predictions: Sequence[str], golds: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-field correctness, keyed by the field name found in the gold answer."""

    totals: dict[str, list[int]] = {}
    for prediction, gold in zip(predictions, golds, strict=True):
        predicted_fields = parse_fields(prediction)
        for name, expected in parse_fields(gold).items():
            hit = totals.setdefault(name, [0, 0])
            hit[1] += 1
            if predicted_fields.get(name) == expected:
                hit[0] += 1
    return {
        name: {
            "correct": hits[0],
            "total": hits[1],
            "accuracy": round(_rate(hits[0], hits[1]), 4),
        }
        for name, hits in sorted(totals.items())
    }



FIELD_LABELS = {
    "sev": ("routine", "watch", "urgent"),
    "cat": ("mobility", "feeding", "respiratory", "skin", "behaviour", "unknown"),
    "act": ("log_only", "recheck_12h", "flag_vet_now"),
}
EXPECTED_FIELDS = tuple(FIELD_LABELS)


def classification_scores(
    predictions: Sequence[str], golds: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per-field accuracy, macro-F1, per-class precision/recall and confusion matrix."""
    if len(predictions) != len(golds):
        raise ValueError("predictions/golds length mismatch")
    parsed_predictions = [parse_fields(value) for value in predictions]
    parsed_golds = [parse_fields(value) for value in golds]
    results: dict[str, dict[str, Any]] = {}
    for name, known_labels in FIELD_LABELS.items():
        expected = [row.get(name, "") for row in parsed_golds]
        actual = [row.get(name, "") for row in parsed_predictions]
        labels = list(known_labels)
        if any(value not in labels for value in expected + actual):
            labels.append("__invalid__")
        matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
        for gold, pred in zip(expected, actual, strict=True):
            matrix[gold if gold in matrix else "__invalid__"][
                pred if pred in labels else "__invalid__"
            ] += 1
        per_class = {}
        f1_values = []
        for label in known_labels:
            tp = matrix[label][label]
            predicted_total = sum(matrix[row][label] for row in labels)
            support = sum(matrix[label].values())
            precision = tp / predicted_total if predicted_total else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            f1_values.append(f1)
        correct = sum(gold == pred for gold, pred in zip(expected, actual, strict=True))
        results[name] = {
            "correct": correct,
            "total": len(golds),
            "accuracy": round(correct / len(golds), 4) if golds else 0.0,
            "macro_f1": round(sum(f1_values) / len(f1_values), 4),
            "per_class": per_class,
            "confusion_matrix": matrix,
            "missing_fields": sum(name not in row for row in parsed_predictions),
        }
    return results


def structured_output_scores(predictions: Sequence[str]) -> dict[str, Any]:
    """Strict output-schema validity, including missing, unexpected and duplicate keys."""
    expected_pattern = re.compile(
        r"sev=(?:routine|watch|urgent) "
        r"cat=(?:mobility|feeding|respiratory|skin|behaviour|unknown) "
        r"act=(?:log_only|recheck_12h|flag_vet_now)"
    )
    missing: dict[str, int] = {}
    unexpected: dict[str, int] = {}
    duplicates = 0
    valid = 0
    for prediction in predictions:
        normalized = normalize_answer(prediction)
        matches = list(_FIELD_PAIR.finditer(normalized))
        fields = parse_fields(prediction)
        keys = [match.group(1) for match in matches]
        duplicates += len(keys) - len(set(keys))
        for key in EXPECTED_FIELDS:
            if key not in fields:
                missing[key] = missing.get(key, 0) + 1
        for key in set(fields) - set(EXPECTED_FIELDS):
            unexpected[key] = unexpected.get(key, 0) + 1
        if (
            set(fields) == set(EXPECTED_FIELDS)
            and len(keys) == len(EXPECTED_FIELDS)
            and expected_pattern.fullmatch(normalized)
        ):
            valid += 1
    total = len(predictions)
    return {
        "examples": total,
        "valid": valid,
        "malformed": total - valid,
        "valid_rate": round(valid / total, 4) if total else 0.0,
        "duplicate_keys": duplicates,
        "missing_fields": missing,
        "unexpected_fields": unexpected,
    }


def format_delta(baseline: float, fine_tuned: float) -> str:
    return f"{baseline:.4f} -> {fine_tuned:.4f} ({fine_tuned - baseline:+.4f})"
