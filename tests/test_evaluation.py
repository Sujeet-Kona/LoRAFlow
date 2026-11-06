"""Evaluation-stage tests: masking, guarding, and report plumbing.

The scoring functions themselves need weights, so they are exercised by the
documented `loraflow evaluate` run; here only the project-owned logic is tested.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from loraflow.config import AppConfig, DataConfig, EvaluationConfig, TrainingConfig  # noqa: E402
from loraflow.data import format_record  # noqa: E402
from loraflow.evaluation import (  # noqa: E402
    IGNORE_INDEX,
    _answer_labels,
    _build_scoring_row,
    _free_memory,
    _note_text,
    evaluate_config,
)  # noqa: E402


def make_config(tmp_path: Path) -> AppConfig:
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({"instruction": "Label", "input": "note", "output": "sev=watch"}),
        encoding="utf-8",
    )
    return AppConfig(
        data=DataConfig(train_file=train, validation_file=train),
        training=TrainingConfig(base_model="fake/model", output_dir=tmp_path / "run"),
        evaluation=EvaluationConfig(output_file=tmp_path / "report.json"),
    )


def test_scoring_row_rejects_truncated_completion() -> None:
    with pytest.raises(ValueError, match="exceeding max_seq_length=5"):
        _build_scoring_row([1, 2, 3], [4, 5, 6], max_length=5, source_line=17)


def test_scoring_row_preserves_full_completion() -> None:
    input_ids, labels = _build_scoring_row([1, 2], [3, 4], max_length=4, source_line=1)

    assert input_ids == [1, 2, 3, 4]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]


def test_answer_labels_masks_everything_before_the_completion() -> None:
    prompt_ids = [1, 2, 3, 4]
    completion_ids = [5, 6]
    input_ids = prompt_ids + completion_ids

    labels = _answer_labels(prompt_ids, completion_ids, input_ids)

    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 5, 6]


def test_answer_labels_respects_truncation() -> None:
    # input_ids were cut to max_length, so the completion no longer fits.
    labels = _answer_labels([1, 2, 3, 4, 5], [6, 7], [1, 2, 3, 4, 5])

    assert labels == [IGNORE_INDEX] * 5


def test_answer_labels_keeps_only_the_first_completion_token_when_truncated() -> None:
    labels = _answer_labels([1, 2], [3, 4, 5], [1, 2, 3])

    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 3]


def test_note_text_recovers_the_input_block() -> None:
    example = format_record(
        {"instruction": "Label", "input": "Animal A1: lame", "output": "out"},
        line_number=1,
    )

    assert _note_text(example) == "Animal A1: lame"


def test_evaluate_requires_an_existing_adapter(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="Run 'loraflow train' first"):
        evaluate_config(config, adapter_dir=tmp_path / "absent")


def test_evaluate_rejects_negative_generation_count(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config.evaluation.generation_examples = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_config(config, adapter_dir=adapter)


def test_free_memory_does_not_call_mps_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def unexpected_empty_cache() -> None:
        pytest.fail("MPS cache cleanup must not run when MPS is unavailable")

    monkeypatch.setattr(torch.mps, "empty_cache", unexpected_empty_cache)

    _free_memory("cpu")


def test_free_memory_clears_cache_when_mps_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(True))

    _free_memory("mps")

    assert calls == [True]
