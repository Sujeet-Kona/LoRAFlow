"""Train-only TF-IDF + linear classification baseline for structured labels."""

from __future__ import annotations

from typing import Any

from loraflow.data import Example
from loraflow.metrics import (
    FIELD_LABELS,
    classification_scores,
    parse_fields,
    score_generations,
    structured_output_scores,
)

_ACTIONS = {"routine": "log_only", "watch": "recheck_12h", "urgent": "flag_vet_now"}
_REQUIRED = ("sev", "cat")


def _labels(examples: list[Example]) -> dict[str, list[str]]:
    labels = {field: [] for field in _REQUIRED}
    for example in examples:
        parsed = parse_fields(example.completion)
        missing = [field for field in _REQUIRED if field not in parsed]
        if missing:
            raise ValueError(
                f"Line {example.source_line} is missing required baseline labels: {missing}"
            )
        if parsed["sev"] not in _ACTIONS:
            raise ValueError(
                f"Line {example.source_line} has unsupported severity {parsed['sev']!r}"
            )
        if parsed["cat"] not in FIELD_LABELS["cat"]:
            raise ValueError(
                f"Line {example.source_line} has unsupported category {parsed['cat']!r}"
            )
        labels["sev"].append(parsed["sev"])
        labels["cat"].append(parsed["cat"])
    return labels


def evaluate_tfidf_baseline(
    train_examples: list[Example],
    test_examples: list[Example],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit two TF-IDF + LinearSVC classifiers on train only and score test once.

    Hyperparameters are fixed rather than selected against the test set. The
    action field is derived from predicted severity because the benchmark policy
    defines it deterministically; it is not an independent classifier.
    """
    if not train_examples or not test_examples:
        raise ValueError("The TF-IDF baseline needs non-empty train and test splits")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError(
            "The TF-IDF baseline requires scikit-learn; install LoRAFlow with the train extra."
        ) from exc

    train_labels = _labels(train_examples)
    _labels(test_examples)  # Validate held-out targets before scoring.
    train_texts = [_note_text(example) for example in train_examples]
    test_texts = [_note_text(example) for example in test_examples]

    predictions_by_field: dict[str, list[str]] = {}
    for field in _REQUIRED:
        if len(set(train_labels[field])) < 2:
            raise ValueError(
                f"Training split needs at least two classes for {field!r}; "
                "the classical baseline cannot be fitted."
            )
        classifier = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed),
                ),
            ]
        )
        classifier.fit(train_texts, train_labels[field])
        predictions_by_field[field] = classifier.predict(test_texts).tolist()

    predictions = [
        " ".join(
            (
                f"sev={severity}",
                f"cat={category}",
                f"act={_ACTIONS[severity]}",
            )
        )
        for severity, category in zip(
            predictions_by_field["sev"], predictions_by_field["cat"], strict=True
        )
    ]
    golds = [example.completion for example in test_examples]
    return {
        "model": "TF-IDF + LinearSVC",
        "fit_scope": "training split only",
        "hyperparameters": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "seed": seed,
        },
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "per_field": classification_scores(predictions, golds),
        **score_generations(predictions, golds).to_dict(),
        "structured_output": structured_output_scores(predictions),
        "action_policy": "derived from predicted severity; not independently classified",
        "predictions": predictions,
    }


def _note_text(example: Example) -> str:
    """Extract only the observation text; never expose target labels to features."""
    return example.prompt.split("### Input:")[-1].split("### Response:")[0].strip()
