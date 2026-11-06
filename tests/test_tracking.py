"""Tests for optional experiment-tracking adapters."""

from pathlib import Path

from loraflow.tracking import MLflowTracker, NoopTracker


def test_new_structured_metrics_are_logged_as_metrics() -> None:
    from loraflow.tracking import _metrics_only

    metrics = _metrics_only(
        {
            "eval.fine_tuned.per_field.sev.macro_f1": 0.75,
            "eval.fine_tuned.per_field.sev.per_class.watch.precision": 0.8,
            "eval.fine_tuned.per_field.sev.per_class.watch.recall": 0.7,
            "eval.fine_tuned.structured_output.valid": 18,
            "eval.fine_tuned.structured_output.malformed": 6,
        }
    )

    assert metrics == {
        "eval.fine_tuned.per_field.sev.macro_f1": 0.75,
        "eval.fine_tuned.per_field.sev.per_class.watch.precision": 0.8,
        "eval.fine_tuned.per_field.sev.per_class.watch.recall": 0.7,
        "eval.fine_tuned.structured_output.valid": 18.0,
        "eval.fine_tuned.structured_output.malformed": 6.0,
    }


def test_noop_tracker_accepts_export_reports() -> None:
    NoopTracker().log_export({"kind": "adapter", "path": "unused"})


def test_mlflow_tracker_logs_export_metadata_and_manifest(tmp_path: Path, monkeypatch) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    manifest = export_dir / "export_manifest.json"
    manifest.write_text('{"kind":"adapter"}', encoding="utf-8")

    tracker = object.__new__(MLflowTracker)
    calls: dict[str, object] = {}
    monkeypatch.setattr(tracker, "start", lambda: calls.update(started=True))
    monkeypatch.setattr(
        tracker,
        "_log_flat",
        lambda payload, *, prefix="": calls.update(payload=payload, prefix=prefix),
    )
    monkeypatch.setattr(tracker, "log_artifact", lambda path: calls.update(manifest=Path(path)))

    report = {"kind": "adapter", "path": str(export_dir)}
    tracker.log_export(report)

    assert calls["started"] is True
    assert calls["payload"] == report
    assert calls["prefix"] == "export"
    assert calls["manifest"] == manifest
