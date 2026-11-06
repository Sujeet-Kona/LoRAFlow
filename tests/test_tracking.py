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


def test_mlflow_run_id_survives_finish() -> None:
    tracker = object.__new__(MLflowTracker)
    tracker._run = None
    tracker._last_run_id = "completed-run-id"

    assert tracker.run_id == "completed-run-id"


def test_mlflow_keys_sanitize_unicode_and_punctuation() -> None:
    from loraflow.tracking import _safe_mlflow_key

    original = "eval.fine_tuned.structured_output.invalid_values.cat.呼吸相关+不明+嘘气_now_action=flag_vet_now_no_action_re"
    sanitized = _safe_mlflow_key(original)

    assert sanitized.isascii()
    assert all(char.isalnum() or char in "_.:/ -" for char in sanitized)
    assert sanitized.endswith(
        "_" + __import__("hashlib").sha1(original.encode("utf-8")).hexdigest()[:8]
    )
    assert _safe_mlflow_key(original) == sanitized


def test_distinct_unicode_mlflow_keys_do_not_collapse() -> None:
    from loraflow.tracking import _safe_mlflow_key

    first = _safe_mlflow_key("eval.invalid_values.cat.呼吸")
    second = _safe_mlflow_key("eval.invalid_values.cat.咳嗽")

    assert first != second
