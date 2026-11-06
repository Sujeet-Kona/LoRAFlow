"""Dataset loading, instruction formatting and splitting.

This module deliberately depends on the standard library only so dataset
checks can run without torch/transformers installed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n{input_block}### Response:\n"

_INPUT_BLOCK_TEMPLATE = "### Input:\n{input}\n\n"
_BLANK_LINES = re.compile(r"\n{3,}")


def load_jsonl(path: str | Path, text_field: str = "text") -> list[dict[str, Any]]:
    """Load JSONL records and validate that each has a non-empty text field."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    records: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")

            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Line {line_number} must contain a non-empty string in field {text_field!r}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"Dataset contains no usable records: {dataset_path}")

    return records


@dataclass(frozen=True)
class Example:
    """One formatted training or evaluation record.

    ``prompt`` is what the model is conditioned on, ``completion`` is the
    expected continuation, and ``text`` is the concatenation used for training.
    """

    prompt: str
    completion: str
    source_line: int

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.completion}"


def render_prompt(
    instruction: str,
    *,
    input_text: str = "",
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Format one instruction into a prompt string.

    ``{input_block}`` expands to an ``### Input:`` section only when
    ``input_text`` is non-empty, so single-field instructions stay tidy.
    """

    input_block = _INPUT_BLOCK_TEMPLATE.format(input=input_text) if input_text else ""
    try:
        rendered = template.format(instruction=instruction, input_block=input_block)
    except KeyError as exc:
        raise ValueError(
            f"Prompt template references unknown placeholder {exc.args[0]!r}; "
            "only 'instruction' and 'input_block' are supported"
        ) from exc
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid prompt template: {template!r}") from exc

    cleaned = _BLANK_LINES.sub("\n\n", rendered)
    if cleaned and not cleaned[-1].isspace():
        # Guarantee the completion starts on its own line, but never rewrite a
        # template that already ends in whitespace.
        cleaned += "\n"
    return cleaned


def format_record(
    record: Mapping[str, Any],
    *,
    line_number: int = 0,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Example:
    """Convert a raw record into an :class:`Example`, or raise with the line."""

    def require_text(name: str) -> str:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number} must contain a non-empty string in field {name!r}"
            )
        return value.strip()

    if instruction_field not in record and "text" in record:
        # Plain-text records are valid training data but cannot be scored
        # against a held-out completion.
        raise ValueError(
            f"Line {line_number} has no {instruction_field!r} field; only plain "
            f"'text' records were found. Instruction tuning needs "
            f"{instruction_field!r} and {output_field!r}."
        )

    instruction = require_text(instruction_field)
    output = require_text(output_field)
    raw_input = record.get(input_field, "")
    if raw_input is None:
        raw_input = ""
    if not isinstance(raw_input, str):
        raise ValueError(f"Line {line_number} field {input_field!r} must be a string")

    return Example(
        prompt=render_prompt(instruction, input_text=raw_input.strip(), template=template),
        completion=output,
        source_line=line_number,
    )


def load_examples(
    path: str | Path,
    *,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    template: str = DEFAULT_PROMPT_TEMPLATE,
    max_records: int | None = None,
) -> list[Example]:
    """Load and format instruction records from a JSONL file.

    ``max_records`` caps how many usable examples are read, which keeps smoke
    runs fast without editing the data file.
    """

    records = load_jsonl(path, text_field=instruction_field)
    examples: list[Example] = []
    for line_number, record in enumerate(records, start=1):
        examples.append(
            format_record(
                record,
                line_number=line_number,
                instruction_field=instruction_field,
                input_field=input_field,
                output_field=output_field,
                template=template,
            )
        )
        if max_records is not None and len(examples) >= max_records:
            break

    if not examples:
        raise ValueError(f"Dataset contains no usable records: {path}")
    return examples


def split_examples(
    examples: Sequence[Example],
    *,
    validation_size: int | float,
    seed: int,
    minimum_train: int = 1,
) -> tuple[list[Example], list[Example]]:
    """Deterministically partition examples into train and validation halves.

    ``validation_size`` is a fraction when ``< 1`` or an absolute count otherwise.
    The same seed always produces the same split, which is what makes the
    baseline-versus-fine-tuned comparison reproducible.
    """

    total = len(examples)
    if total == 0:
        raise ValueError("Cannot split an empty dataset")
    if minimum_train < 1:
        raise ValueError("minimum_train must be at least one")
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")

    if validation_size < 1:
        val_count = int(total * validation_size)
    else:
        val_count = int(validation_size)

    if val_count > total - minimum_train:
        raise ValueError(
            f"Requested {val_count} validation examples from {total}; "
            f"at least {minimum_train} training examples must remain"
        )
    if val_count < 1:
        raise ValueError(
            f"Dataset of {total} examples cannot reserve a validation split "
            f"with at least {minimum_train} training example"
        )

    order = list(range(total))
    random.Random(seed).shuffle(order)
    validation_indexes = set(order[:val_count])
    train = [ex for i, ex in enumerate(examples) if i not in validation_indexes]
    validation = [ex for i, ex in enumerate(examples) if i in validation_indexes]
    return train, validation


def deduplicate(examples: Iterable[Example]) -> list[Example]:
    """Drop later examples whose full text already appeared earlier."""

    seen: set[str] = set()
    kept: list[Example] = []
    for example in examples:
        if example.text in seen:
            continue
        seen.add(example.text)
        kept.append(example)
    return kept


def dataset_report(examples: Sequence[Example]) -> dict[str, Any]:
    """Summarise a split for logging; contains no model-dependent numbers."""

    lengths = [len(example.text) for example in examples]
    return {
        "examples": len(examples),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "unique_texts": len({example.text for example in examples}),
    }


def file_identity(path: str | Path) -> dict[str, Any]:
    """Content identity of a dataset file, so a run can be tied to its exact corpus."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {target}")
    payload = target.read_bytes()
    return {
        "path": str(target),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "records": sum(1 for line in payload.decode("utf-8").splitlines() if line.strip()),
    }


def dataset_provenance(
    train_file: str | Path, validation_file: str | Path | None
) -> dict[str, Any]:
    """Identify the files a split actually came from.

    ``held_out_source`` distinguishes a real held-out file from a slice carved out of
    the training file, which is the difference between a test set and a checkpoint
    selector.
    """

    held_out = Path(validation_file) if validation_file is not None else Path(train_file)
    return {
        "train": file_identity(train_file),
        "held_out": file_identity(held_out),
        "held_out_source": "validation_file"
        if validation_file is not None
        else "carved_from_train",
    }
