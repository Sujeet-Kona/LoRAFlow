"""Reproducible multi-seed experiments for the committed Prompt-v2 benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = (42, 43, 44)
TRACKS = ("a", "b", "c")
METRICS = (
    "exact_match",
    "prefix_match",
    "mean_answer_loss",
    "perplexity",
    "answer_token_accuracy",
    "category_macro_f1",
    "severity_macro_f1",
    "structured_valid_rate",
    "classical_exact_match",
)


def _metric_values(report: dict[str, Any]) -> dict[str, float | None]:
    tuned = report["fine_tuned"]
    fields = tuned.get("per_field", {})
    classical = report.get("classical_baseline")
    values: dict[str, float | None] = {
        "exact_match": tuned.get("exact_match"),
        "prefix_match": tuned.get("prefix_match"),
        "mean_answer_loss": tuned.get("mean_answer_loss"),
        "perplexity": tuned.get("perplexity"),
        "answer_token_accuracy": tuned.get("answer_token_accuracy"),
        "category_macro_f1": fields.get("cat", {}).get("macro_f1"),
        "severity_macro_f1": fields.get("sev", {}).get("macro_f1"),
        "structured_valid_rate": tuned.get("structured_output", {}).get("valid_rate"),
        "classical_exact_match": (
            classical.get("exact_match") if isinstance(classical, dict) else None
        ),
    }
    return values


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize seed-level scores without treating examples as independent runs."""
    if not reports:
        raise ValueError("At least one completed seed report is required")

    per_seed = []
    for report in reports:
        seed = report.get("reproducibility", {}).get("seed")
        if not isinstance(seed, int):
            raise ValueError("Each evaluation report must record an integer seed")
        per_seed.append({"seed": seed, **_metric_values(report)})

    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = [row[metric] for row in per_seed if row[metric] is not None]
        if not values:
            summary[metric] = None
            continue
        numeric = [float(value) for value in values]
        summary[metric] = {
            "n": len(numeric),
            "mean": round(statistics.fmean(numeric), 6),
            "std": round(statistics.stdev(numeric), 6) if len(numeric) > 1 else 0.0,
            "min": round(min(numeric), 6),
            "max": round(max(numeric), 6),
        }

    return {
        "protocol": {
            "aggregation_unit": "training seed",
            "confidence_intervals": "not reported; seed count is intentionally small",
            "test_split": "same fixed committed test split for every seed",
            "interpretation": (
                "Descriptive sensitivity analysis only. Seeds are not independent "
                "datasets and do not establish broad generalization."
            ),
        },
        "seeds": per_seed,
        "summary": summary,
    }


def _verify_data() -> None:
    subprocess.run(
        ["python", "scripts/verify_prompt_v2_data.py"],
        cwd=ROOT,
        check=True,
    )


def run_seed_sweep(track: str, seeds: list[int], output_root: Path) -> dict[str, Any]:
    track = track.lower()
    if track not in TRACKS:
        raise ValueError(f"track must be one of {', '.join(TRACKS)}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide one or more unique seeds")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers")

    config_source = ROOT / "configs" / f"v4_prompt_v2_track_{track}.yaml"
    base_config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    output_root = output_root if output_root.is_absolute() else ROOT / output_root
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    for seed in seeds:
        run_dir = output_root / f"track-{track}" / f"seed-{seed}"
        config_path = config_dir / f"track-{track}-seed-{seed}.yaml"
        report_path = run_dir / "evaluation_report.json"
        export_dir = run_dir / "exported"
        if run_dir.exists() or config_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite seed {seed} outputs: {run_dir} or {config_path}"
            )
        planned.append((seed, run_dir, config_path, report_path, export_dir))

    _verify_data()
    reports = []
    for seed, run_dir, config_path, report_path, export_dir in planned:
        config = json.loads(json.dumps(base_config))
        config["training"]["seed"] = seed
        config["training"]["output_dir"] = str(run_dir / "training")
        config["training"]["run_name"] = f"prompt-v2-track-{track}-seed-{seed}"
        config["evaluation"]["output_file"] = str(report_path)
        config["export"]["output_dir"] = str(export_dir)
        config["export"]["merged_dir"] = str(export_dir / "merged")
        config["tracking"]["run_name"] = f"prompt-v2-track-{track}-seed-{seed}"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        for command in (
            ["loraflow", "train", "--config", str(config_path)],
            ["loraflow", "evaluate", "--config", str(config_path)],
            ["loraflow", "export", "--config", str(config_path)],
        ):
            subprocess.run(command, cwd=ROOT, check=True)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        recorded_seed = report.get("reproducibility", {}).get("seed")
        if recorded_seed != seed:
            raise RuntimeError(
                f"Expected seed {seed} in {report_path}, found {recorded_seed!r}"
            )
        reports.append(report)

    result = summarize_reports(reports)
    result["track"] = track.upper()
    result["test_dataset"] = reports[0].get("held_out_dataset")
    result["config_files"] = [str(item[2].relative_to(ROOT)) for item in planned]
    summary_path = output_root / f"track-{track}-summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Multi-seed summary written to {summary_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, default="c")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/v4-prompt-v2/seed-sweep"),
    )
    args = parser.parse_args()
    run_seed_sweep(args.track, args.seeds, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
