"""Dataset loading and validation helpers."""

import json
from pathlib import Path
from typing import Any


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
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")

            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Line {line_number} must contain a non-empty string "
                    f"in field {text_field!r}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"Dataset contains no usable records: {dataset_path}")

    return records
