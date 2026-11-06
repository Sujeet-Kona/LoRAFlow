from __future__ import annotations

import pytest

from loraflow.concept_audit import audit_report


def _report() -> dict:
    return {
        "reproducibility": {"seed": 44},
        "examples": [
            {
                "input": "Observation: reduced walking distance. neutral. stable.",
                "gold": "sev=watch cat=mobility act=recheck_12h",
                "fine_tuned": "sev=watch cat=unknown act=recheck_12h",
            },
            {
                "input": "Observation: reduced walking distance. neutral. stable.",
                "gold": "sev=watch cat=mobility act=recheck_12h",
                "fine_tuned": "sev=watch cat=mobility act=recheck_12h",
            },
        ],
    }


def test_concept_audit_reports_per_concept_accuracy_and_predictions() -> None:
    result = audit_report(_report())
    assert result["seed"] == 44
    assert result["generation_examples"] == 2
    assert result["concepts"]["reduced walking distance"] == {
        "category": "mobility",
        "n": 2,
        "category_accuracy": 0.5,
        "severity_accuracy": 1.0,
        "exact_match": 0.5,
        "predicted_categories": {"mobility": 1, "unknown": 1},
        "predicted_severities": {"watch": 2},
    }


def test_concept_audit_rejects_category_mismatch_in_gold() -> None:
    report = _report()
    report["examples"][0]["gold"] = "sev=watch cat=skin act=recheck_12h"
    with pytest.raises(ValueError, match="does not match gold category"):
        audit_report(report)


def test_concept_audit_rejects_unknown_or_ambiguous_concepts() -> None:
    report = _report()
    report["examples"][0]["input"] = "Observation: symptom not in inventory. neutral. stable."
    with pytest.raises(ValueError, match="exactly one known concept"):
        audit_report(report)


def test_concept_audit_requires_generated_examples() -> None:
    with pytest.raises(ValueError, match="no generated examples"):
        audit_report({"examples": []})
