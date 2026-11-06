"""Tests for the train-only classical benchmark baseline."""

import pytest

pytest.importorskip("sklearn")

from loraflow.baselines import evaluate_tfidf_baseline  # noqa: E402
from loraflow.data import Example  # noqa: E402


def _example(note: str, severity: str, category: str, line: int) -> Example:
    action = {"routine": "log_only", "watch": "recheck_12h", "urgent": "flag_vet_now"}[severity]
    return Example(
        prompt=f"### Instruction:\nLabel\n### Input:\n{note}\n\n### Response:\n",
        completion=f"sev={severity} cat={category} act={action}",
        source_line=line,
    )


def test_tfidf_baseline_fits_train_and_scores_held_out() -> None:
    train = [
        _example("steady walking and normal gait", "routine", "mobility", 1),
        _example("normal appetite and feed intake", "routine", "feeding", 2),
        _example("repeated stumbling and worsening gait", "watch", "mobility", 3),
        _example("reduced feed intake and slow eating", "watch", "feeding", 4),
        _example("unable to stand and critical condition", "urgent", "mobility", 5),
        _example("refuses all feed and severe decline", "urgent", "feeding", 6),
    ]
    test = [
        _example("stumbling with reduced movement", "watch", "mobility", 7),
        _example("critical refusal of food", "urgent", "feeding", 8),
    ]

    report = evaluate_tfidf_baseline(train, test, seed=7)

    assert report["model"] == "TF-IDF + LinearSVC"
    assert report["fit_scope"] == "training split only"
    assert report["train_examples"] == 6
    assert report["test_examples"] == 2
    assert len(report["predictions"]) == 2
    assert set(report["per_field"]) == {"sev", "cat", "act"}
    assert report["structured_output"]["valid"] == 2
    assert report["action_policy"].startswith("derived from predicted severity")


def test_tfidf_baseline_rejects_missing_target_labels() -> None:
    train = [_example("steady gait", "routine", "mobility", 1)]
    bad_test = [Example(prompt="### Input:\nunknown\n", completion="free text", source_line=2)]

    with pytest.raises(ValueError, match="missing required baseline labels"):
        evaluate_tfidf_baseline(train, bad_test)
