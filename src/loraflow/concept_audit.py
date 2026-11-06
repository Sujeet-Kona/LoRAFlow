"""Concept-level diagnostics for generated Prompt-v2 evaluation reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loraflow.metrics import parse_fields
from loraflow.v4_benchmarks import CONCEPTS


def audit_report(report: dict[str, Any], report_name: str = "<memory>") -> dict[str, Any]:
    """Group held-out generation errors by the exact synthetic symptom concept."""
    examples = report.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{report_name}: evaluation report has no generated examples")

    grouped: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        note = example.get("input")
        if not isinstance(note, str):
            raise ValueError(f"{report_name}: example {index} has no input text")

        matches = [
            (category, phrase)
            for category, phrases in CONCEPTS.items()
            for phrase in phrases
            if phrase in note
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{report_name}: example {index} must contain exactly one known concept; "
                f"found {[phrase for _, phrase in matches]}"
            )
        category, concept = matches[0]

        gold = parse_fields(str(example.get("gold", "")))
        predicted = parse_fields(str(example.get("fine_tuned", "")))
        if gold.get("cat") != category:
            raise ValueError(
                f"{report_name}: example {index} concept category {category!r} "
                f"does not match gold category {gold.get('cat')!r}"
            )

        stats = grouped.setdefault(
            concept,
            {
                "category": category,
                "n": 0,
                "category_correct": 0,
                "severity_correct": 0,
                "exact_match": 0,
                "predicted_categories": Counter(),
                "predicted_severities": Counter(),
            },
        )
        stats["n"] += 1
        stats["category_correct"] += predicted.get("cat") == gold.get("cat")
        stats["severity_correct"] += predicted.get("sev") == gold.get("sev")
        stats["exact_match"] += all(
            predicted.get(key) == gold.get(key) for key in ("sev", "cat", "act")
        )
        stats["predicted_categories"][predicted.get("cat", "<missing>")] += 1
        stats["predicted_severities"][predicted.get("sev", "<missing>")] += 1

    concepts = {}
    for concept, stats in sorted(grouped.items()):
        count = stats["n"]
        concepts[concept] = {
            "category": stats["category"],
            "n": count,
            "category_accuracy": round(stats["category_correct"] / count, 4),
            "severity_accuracy": round(stats["severity_correct"] / count, 4),
            "exact_match": round(stats["exact_match"] / count, 4),
            "predicted_categories": dict(sorted(stats["predicted_categories"].items())),
            "predicted_severities": dict(sorted(stats["predicted_severities"].items())),
        }

    return {
        "report": report_name,
        "seed": report.get("reproducibility", {}).get("seed"),
        "generation_examples": len(examples),
        "concepts": concepts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        results.append(audit_report(report, str(path)))

    payload = {"diagnostic": "concept-level held-out generation performance", "reports": results}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote concept-level report to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
