from pathlib import Path

import pytest

from loraflow.data import load_jsonl


def test_load_jsonl_reads_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"text": "first example"}\n\n{"text": "second example"}\n',
        encoding="utf-8",
    )

    records = load_jsonl(path)

    assert len(records) == 2
    assert records[0]["text"] == "first example"


def test_load_jsonl_rejects_missing_text_field(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        load_jsonl(path)


def test_load_jsonl_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)
