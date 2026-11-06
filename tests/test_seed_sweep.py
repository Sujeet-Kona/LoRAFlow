"""Tests for multi-seed evaluation aggregation."""

import pytest

from loraflow.seed_sweep import summarize_reports


def make_report(seed: int, exact: float, category_f1: float) -> dict:
    return {
        "reproducibility": {"seed": seed},
        "fine_tuned": {
            "exact_match": exact,
            "prefix_match": exact,
            "mean_answer_loss": 0.1,
            "perplexity": 1.1,
            "answer_token_accuracy": 0.9,
            "per_field": {
                "cat": {"macro_f1": category_f1},
                "sev": {"macro_f1": 0.8},
            },
            "structured_output": {"valid_rate": 1.0},
        },
        "classical_baseline": {"exact_match": 0.2},
    }


def test_summarize_reports_aggregates_across_seeds() -> None:
    result = summarize_reports([make_report(42, 0.4, 0.3), make_report(43, 0.6, 0.5)])

    assert [row["seed"] for row in result["seeds"]] == [42, 43]
    assert result["summary"]["exact_match"] == {
        "n": 2,
        "mean": 0.5,
        "std": pytest.approx(0.141421, abs=1e-6),
        "min": 0.4,
        "max": 0.6,
    }
    assert result["summary"]["category_macro_f1"]["mean"] == 0.4
    assert result["protocol"]["confidence_intervals"].startswith("not reported")


def test_summarize_reports_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_reports([])


def test_summarize_reports_requires_seed_provenance() -> None:
    report = make_report(42, 0.4, 0.3)
    report["reproducibility"] = {}

    with pytest.raises(ValueError, match="integer seed"):
        summarize_reports([report])


def test_summarize_reports_preserves_missing_classical_baseline() -> None:
    report = make_report(42, 0.4, 0.3)
    report["classical_baseline"] = None

    result = summarize_reports([report])

    assert result["summary"]["classical_exact_match"] is None
