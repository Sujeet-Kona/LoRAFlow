"""Export validation tests. These exercise the file-level checks only, so no
weights are loaded."""

import json
from pathlib import Path

import pytest

pytest.importorskip("peft")

from loraflow.config import AppConfig, DataConfig, ExportConfig, TrainingConfig  # noqa: E402
from loraflow.export import (  # noqa: E402
    ADAPTER_FILES,
    ExportValidationError,
    _max_abs_delta,
    _save_dtype,
    validate_export,
)
from loraflow.training import ADAPTER_DIRNAME  # noqa: E402


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(train_file=tmp_path / "train.jsonl"),
        training=TrainingConfig(base_model="fake/model", output_dir=tmp_path / "run"),
        export=ExportConfig(output_dir=tmp_path / "exported"),
    )


def write_minimal_adapter(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}), encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(b"pretend-weights")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_validate_export_rejects_missing_directory(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(ExportValidationError, match="not a directory"):
        validate_export(config, tmp_path / "nope", kind="adapter")


def test_validate_export_lists_missing_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExportValidationError, match="missing"):
        validate_export(config, directory, kind="adapter")


def test_validate_export_rejects_zero_byte_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    directory = write_minimal_adapter(tmp_path / "adapter")
    (directory / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(ExportValidationError, match="zero-byte"):
        validate_export(config, directory, kind="adapter")


def test_validate_export_rejects_unknown_kind(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    directory = write_minimal_adapter(tmp_path / "adapter")

    with pytest.raises(KeyError):
        validate_export(config, directory, kind="mystery")


def test_required_adapter_files_are_named() -> None:
    # The trainer writes exactly these; the guard would be meaningless otherwise.
    assert ADAPTER_FILES == ("adapter_config.json", "adapter_model.safetensors")


def test_export_adapter_reports_missing_source(tmp_path: Path) -> None:
    from loraflow.export import export_adapter

    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="Run 'loraflow train' first"):
        export_adapter(config, adapter_dir=tmp_path / "absent")


def test_export_adapter_uses_run_directory_convention(tmp_path: Path) -> None:
    from loraflow.export import _resolve_adapter_dir

    config = make_config(tmp_path)
    expected = Path(config.training.output_dir) / ADAPTER_DIRNAME
    expected.mkdir(parents=True)

    assert _resolve_adapter_dir(config, None) == expected


def test_max_abs_delta_compares_tensors() -> None:
    torch = pytest.importorskip("torch")

    assert _max_abs_delta(torch.zeros(3), torch.ones(3)) == pytest.approx(1.0)


def test_max_abs_delta_rejects_shape_change() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ExportValidationError, match="Logit shape changed"):
        _max_abs_delta(torch.zeros(3), torch.zeros(4))


def test_save_dtype_mapping() -> None:
    torch = pytest.importorskip("torch")

    assert _save_dtype("same") == torch.float32
    assert _save_dtype("bf16") == torch.bfloat16
    assert _save_dtype("fp16") == torch.float16


def test_save_dtype_rejects_unknown_name() -> None:
    pytest.importorskip("torch")

    with pytest.raises(KeyError):
        _save_dtype("int4")
