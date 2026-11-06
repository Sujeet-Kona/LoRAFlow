import math

import pytest

from loraflow.metrics import (
    exact_match,
    format_delta,
    normalize_answer,
    perplexity_from_loss,
    prefix_match,
    score_generations,
    token_accuracy,
)


def test_normalize_answer_strips_scaffolding() -> None:
    text = "### Response:\n```text\nsev=Watch  cat=feeding\n```\n"

    assert normalize_answer(text) == "sev=watch cat=feeding"


def test_normalize_answer_collapses_whitespace_and_trailing_punct() -> None:
    assert normalize_answer("  a   b  c...  ") == "a b c"
    assert normalize_answer("Line one\n\nLine two!") == "line one line two"


def test_exact_match_ignores_case_and_punctuation() -> None:
    assert exact_match("SEV=urgent cat=skin\n", "sev=urgent cat=skin")


def test_exact_match_rejects_different_labels() -> None:
    assert not exact_match("sev=urgent cat=skin", "sev=urgent cat=behaviour")


def test_prefix_match_detects_right_direction() -> None:
    assert prefix_match("alpha beta gamma extra words here", "alpha beta gamma")
    assert not prefix_match("alpha delta gamma", "alpha beta gamma")


def test_prefix_match_handles_short_gold() -> None:
    assert prefix_match("alpha beta", "alpha", tokens=3)
    assert not prefix_match("beta", "", tokens=3)


def test_score_generations_counts_hits() -> None:
    preds = ["sev=watch cat=skin act=recheck_12h", "wrong answer"]
    golds = ["SEV=watch cat=skin act=recheck_12h", "sev=routine cat=skin act=log_only"]

    scores = score_generations(preds, golds)

    assert scores.examples == 2
    assert scores.exact_match == 0.5
    assert scores.prefix_match == 0.5
    assert scores.to_dict()["exact_match"] == 0.5


def test_score_generations_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        score_generations(["a"], [])


def test_score_generations_empty_is_zero_not_crash() -> None:
    scores = score_generations([], [])
    assert scores.examples == 0
    assert scores.exact_match == 0.0


def test_perplexity_from_loss() -> None:
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(2.0), math.e**2)


def test_perplexity_rejects_negative_loss() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        perplexity_from_loss(-0.1)


def test_token_accuracy_skips_masked_positions() -> None:
    # Position 0 has no predecessor, so it is never counted.
    labels = [-100, 5, 6, 7]
    predicted = [0, 5, 9, 7]

    correct, total = token_accuracy(predicted, labels)

    assert (correct, total) == (2, 3)


def test_token_accuracy_all_masked_returns_zero() -> None:
    assert token_accuracy([1, 2], [-100, -100]) == (0, 0)


def test_parse_fields_extracts_key_value_pairs() -> None:
    from loraflow.metrics import parse_fields

    assert parse_fields("sev=watch cat=skin act=recheck_12h") == {
        "sev": "watch",
        "cat": "skin",
        "act": "recheck_12h",
    }
    # A rambling base model still yields parseable fields.
    assert parse_fields("The label is sev=urgent cat=skin and then more text") == {
        "sev": "urgent",
        "cat": "skin",
    }


def test_field_scores_counts_per_field_correctness() -> None:
    from loraflow.metrics import field_scores

    preds = ["sev=watch cat=skin act=log_only", "sev=watch cat=behaviour act=recheck_12h"]
    golds = ["sev=watch cat=skin act=recheck_12h", "sev=watch cat=skin act=recheck_12h"]

    scores = field_scores(preds, golds)

    assert scores["sev"]["accuracy"] == 1.0
    assert scores["cat"]["correct"] == 1 and scores["cat"]["total"] == 2
    assert scores["act"]["accuracy"] == 0.5


def test_field_scores_empty_inputs() -> None:
    from loraflow.metrics import field_scores

    assert field_scores([], []) == {}


def test_classification_scores_include_per_class_metrics_and_confusion_matrix() -> None:
    from loraflow.metrics import classification_scores

    scores = classification_scores(
        ["sev=watch cat=skin act=recheck_12h", "sev=urgent cat=skin act=flag_vet_now"],
        ["sev=watch cat=skin act=recheck_12h", "sev=routine cat=feeding act=log_only"],
    )

    assert scores["sev"]["accuracy"] == 0.5
    assert scores["sev"]["per_class"]["watch"]["precision"] == 1.0
    assert scores["sev"]["per_class"]["routine"]["recall"] == 0.0
    assert scores["sev"]["confusion_matrix"]["routine"]["urgent"] == 1
    assert "macro_f1" in scores["cat"]


def test_structured_output_scores_detect_malformed_and_missing_fields() -> None:
    from loraflow.metrics import structured_output_scores

    scores = structured_output_scores(
        [
            "sev=watch cat=skin act=recheck_12h",
            "sev=urgent cat=skin",
            "sev=urgent sev=watch cat=skin act=flag_vet_now",
            "answer: sev=watch cat=skin act=recheck_12h",
            "sev=critical cat=skin act=flag_vet_now",
            "sev=watch\ncat=skin act=recheck_12h",
            "sev=watch cat=skin act=recheck_12h extra=foo",
        ]
    )

    assert scores["valid"] == 1
    assert scores["malformed"] == 6
    assert scores["missing_fields"]["act"] == 1
    assert scores["duplicate_keys"] == 1
    assert scores["invalid_values"]["sev"]["critical"] == 1
    assert scores["unexpected_fields"]["extra"] == 1


def test_format_delta_shows_both_ends() -> None:
    assert format_delta(0.1, 0.25) == "0.1000 -> 0.2500 (+0.1500)"


def test_token_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        token_accuracy([1, 2], [-100])
