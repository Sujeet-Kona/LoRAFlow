from pathlib import Path

import pytest

from loraflow.data import (
    DEFAULT_PROMPT_TEMPLATE,
    dataset_report,
    deduplicate,
    format_record,
    load_examples,
    load_jsonl,
    render_prompt,
    split_examples,
)


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


def test_load_jsonl_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "absent.jsonl")


def test_load_jsonl_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no usable records"):
        load_jsonl(path)


def test_render_prompt_includes_input_block() -> None:
    prompt = render_prompt("Label the note", input_text="Animal A1: lame")

    assert "### Instruction:\nLabel the note\n" in prompt
    assert "### Input:\nAnimal A1: lame\n\n" in prompt
    assert prompt.endswith("### Response:\n")


def test_render_prompt_omits_empty_input_block() -> None:
    prompt = render_prompt("Explain LoRA")

    assert "### Input:" not in prompt
    assert prompt == "### Instruction:\nExplain LoRA\n### Response:\n"


def test_render_prompt_honours_custom_template() -> None:
    prompt = render_prompt(
        "Summarise", input_text="text here", template="Q: {instruction}\n{input_block}A: "
    )

    assert prompt == "Q: Summarise\n### Input:\ntext here\n\nA: "


def test_render_prompt_keeps_trailing_whitespace_from_template() -> None:
    # A template ending in a space must not be rewritten; only a template with
    # no trailing whitespace gets a newline boundary.
    assert render_prompt("x", template="A: {instruction} ") == "A: x "
    assert render_prompt("x", template="A: {instruction}") == "A: x\n"


def test_render_prompt_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        render_prompt("x", template="{instruction} {bogus}")


def test_format_record_builds_prompt_and_completion() -> None:
    example = format_record(
        {"instruction": "Label", "input": "note", "output": "sev=watch"},
        line_number=4,
    )

    assert example.completion == "sev=watch"
    assert example.source_line == 4
    assert example.text == example.prompt + example.completion


def test_format_record_reports_the_offending_line() -> None:
    with pytest.raises(ValueError, match="Line 7 .* 'output'"):
        format_record({"instruction": "Label", "input": "note"}, line_number=7)


def test_format_record_rejects_non_string_input() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        format_record({"instruction": "Label", "input": 12, "output": "ok"}, line_number=1)


def test_format_record_explains_plain_text_only_records() -> None:
    with pytest.raises(ValueError, match="Instruction tuning needs"):
        format_record({"text": "just prose"}, line_number=1)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def test_load_examples_reads_and_formats(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            '{"instruction": "Label", "input": "note one", "output": "sev=urgent"}',
            '{"instruction": "Label", "input": "note two", "output": "sev=routine"}',
        ],
    )

    examples = load_examples(path)

    assert len(examples) == 2
    assert examples[1].completion == "sev=routine"


def test_load_examples_respects_max_records(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [f'{{"instruction": "Label", "input": "n{i}", "output": "o{i}"}}' for i in range(10)],
    )

    assert len(load_examples(path, max_records=3)) == 3


def test_load_examples_surfaces_line_number_of_bad_record(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            '{"instruction": "Label", "input": "ok", "output": "fine"}',
            '{"instruction": "Label", "input": "ok"}',
        ],
    )

    with pytest.raises(ValueError, match="Line 2"):
        load_examples(path)


def make_examples(count: int) -> list:
    return [
        format_record(
            {"instruction": "Label", "input": f"note {i}", "output": f"out {i}"},
            line_number=i,
        )
        for i in range(count)
    ]


def test_split_examples_is_deterministic_for_a_seed() -> None:
    examples = make_examples(50)

    train_a, val_a = split_examples(examples, validation_size=0.2, seed=42)
    train_b, val_b = split_examples(examples, validation_size=0.2, seed=42)

    assert [e.text for e in train_a] == [e.text for e in train_b]
    assert [e.text for e in val_a] == [e.text for e in val_b]
    assert len(val_a) == 10
    assert len(train_a) == 40


def test_split_examples_produces_disjoint_halves() -> None:
    train, validation = split_examples(make_examples(30), validation_size=0.25, seed=7)

    train_texts = {e.text for e in train}
    val_texts = {e.text for e in validation}
    assert train_texts.isdisjoint(val_texts)
    assert len(train_texts) + len(val_texts) == 30


def test_split_examples_accepts_absolute_count() -> None:
    train, validation = split_examples(make_examples(20), validation_size=5, seed=1)

    assert len(validation) == 5
    assert len(train) == 15


def test_split_examples_rejects_impossible_split() -> None:
    with pytest.raises(ValueError, match="cannot reserve a validation split"):
        split_examples(make_examples(1), validation_size=0.5, seed=0)


def test_split_examples_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        split_examples([], validation_size=0.1, seed=0)


def test_deduplicate_keeps_first_occurrence() -> None:
    base = make_examples(3)
    duplicated = [base[0], base[1], base[2], base[1], base[0]]

    unique = deduplicate(duplicated)

    assert [e.source_line for e in unique] == [0, 1, 2]
    assert len(unique) == 3


def test_dataset_report_counts_and_bounds() -> None:
    examples = make_examples(4)
    report = dataset_report(examples)

    assert report["examples"] == 4
    assert report["unique_texts"] == 4
    assert report["max_chars"] >= report["min_chars"]
    assert report["mean_chars"] > 0


def test_default_template_matches_config_default() -> None:
    # config.DataConfig repeats the default; keep the two in sync.
    from loraflow.config import DataConfig

    assert DataConfig(train_file="x.jsonl").prompt_template == DEFAULT_PROMPT_TEMPLATE


def test_split_examples_rejects_validation_count_that_leaves_no_train_data() -> None:
    with pytest.raises(ValueError, match="at least 1 training examples must remain"):
        split_examples(make_examples(5), validation_size=5, seed=0)


def test_split_examples_rejects_invalid_minimum_train() -> None:
    with pytest.raises(ValueError, match="minimum_train must be at least one"):
        split_examples(make_examples(5), validation_size=1, seed=0, minimum_train=0)
