"""Summarise completed v4 track evaluations without recomputing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRACKS = ("a", "b", "c")


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _field(block: dict[str, Any], name: str, metric: str) -> Any:
    return block.get("per_field", {}).get(name, {}).get(metric)


def _field_cell(block: dict[str, Any], name: str) -> str:
    accuracy = _field(block, name, "accuracy")
    macro_f1 = _field(block, name, "macro_f1")
    return f"{_percent(accuracy)} / {_percent(macro_f1)}"


def build_summary(artifact_root: Path) -> str:
    rows = []
    provenance = []
    for track in TRACKS:
        run_dir = artifact_root / f"v4-track-{track}-run"
        metadata_path = run_dir / "training_metadata.json"
        report_path = run_dir / "evaluation_report.json"
        if not metadata_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                f"Track {track.upper()} is incomplete; expected {metadata_path} and {report_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("held_out_source") != "test_file":
            raise ValueError(f"Track {track.upper()} did not evaluate its explicit test file")

        base = report["baseline"]
        adapter = report["fine_tuned"]
        classical = report.get("classical_baseline")
        if classical is None:
            raise ValueError(
                f"Track {track.upper()} has no TF-IDF baseline; check the train/test label schema "
                "and install scikit-learn from the train extra."
            )

        rows.append(
            "| {track} | {n} | {loss} | {base_sev} | {base_cat} | {base_em} | "
            "{adapter_sev} | {adapter_cat} | {adapter_valid} | {adapter_em} | "
            "{tfidf_sev} | {tfidf_cat} | {tfidf_valid} | {tfidf_em} |".format(
                track=track.upper(),
                n=report["held_out"]["examples"],
                loss=f"{metadata.get('best_eval_loss', float('nan')):.4f}",
                base_sev=_percent(_field(base, "sev", "accuracy")),
                base_cat=_percent(_field(base, "cat", "accuracy")),
                base_em=_percent(base.get("exact_match")),
                adapter_sev=_field_cell(adapter, "sev"),
                adapter_cat=_field_cell(adapter, "cat"),
                adapter_valid=_percent(adapter.get("structured_output", {}).get("valid_rate")),
                adapter_em=_percent(adapter.get("exact_match")),
                tfidf_sev=_field_cell(classical, "sev"),
                tfidf_cat=_field_cell(classical, "cat"),
                tfidf_valid=_percent(classical.get("structured_output", {}).get("valid_rate")),
                tfidf_em=_percent(classical.get("exact_match")),
            )
        )
        test_identity = report.get("held_out_dataset", {})
        provenance.append(
            f"- Track {track.upper()} test SHA-256: "
            f"`{test_identity.get('sha256', 'not recorded')}`; "
            f"training run ID: `{metadata.get('mlflow_run_id') or 'not recorded'}`."
        )

    table = [
        "| Track | Test n | Best val loss | Base sev acc | Base cat acc | Base EM | "
        "LoRA sev acc / F1 | LoRA cat acc / F1 | LoRA valid | LoRA EM | "
        "TF-IDF sev acc / F1 | TF-IDF cat acc / F1 | TF-IDF valid | TF-IDF EM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]
    return "\n".join(
        [
            "# LoRAFlow v4 controlled benchmark results",
            "",
            *table,
            "",
            "Accuracy, macro-F1, exact match (EM), and strict structured-output validity are measured on each explicit test split. The TF-IDF vocabulary and classifiers are fit on training data only.",
            "",
            "Action is derived deterministically from severity and is not independent model evidence.",
            "",
            "This is a small curated synthetic engineering benchmark, not veterinary guidance, clinical decision support, or evidence of broad natural-language generalization.",
            "",
            "## Reproducibility",
            "",
            *provenance,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v4-summary.md"))
    args = parser.parse_args()
    summary = build_summary(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
