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
    _ensure_separate_export_path,
    _prepare_export_target,
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


def test_export_free_memory_does_not_call_unavailable_mps(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from loraflow.export import _free_memory

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def unexpected_empty_cache() -> None:
        pytest.fail("MPS cache cleanup must not run when MPS is unavailable")

    monkeypatch.setattr(torch.mps, "empty_cache", unexpected_empty_cache)
    _free_memory("mps")


def test_export_free_memory_clears_available_mps(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from loraflow.export import _free_memory

    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(True))
    _free_memory("mps")
    assert calls == [True]


def test_export_destination_cannot_overlap_source(tmp_path: Path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()

    for target in (source, tmp_path, source / "nested"):
        with pytest.raises(ValueError, match="overlaps source adapter"):
            _ensure_separate_export_path(source, target)

    assert source.is_dir()



def test_export_target_refuses_to_replace_existing_data_by_default(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _prepare_export_target(target, overwrite=False)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_export_target_replaces_existing_directory_only_when_requested(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    _prepare_export_target(target, overwrite=True)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_export_target_replaces_existing_file_only_when_requested(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.write_text("old", encoding="utf-8")

    _prepare_export_target(target, overwrite=True)

    assert target.is_dir()


def test_adapter_export_reload_loads_base_and_applies_adapter(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from peft import PeftModel
    from safetensors import torch as safetensors_torch
    import loraflow.export as export_module

    config = make_config(tmp_path)
    directory = write_minimal_adapter(tmp_path / "adapter")
    calls = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self):
            return 10

        def __call__(self, prompt, *, return_tensors):
            assert prompt == "LoRAFlow adapter reload validation."
            assert return_tensors == "pt"
            return {"input_ids": torch.ones((1, 3), dtype=torch.long)}

    class FakeModel:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

        def __call__(self, **inputs):
            assert inputs["input_ids"].shape == (1, 3)
            calls.append(("forward",))

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path):
            assert adapter_path == str(directory)
            calls.append(("adapter",))
            return FakeModel()

    monkeypatch.setattr(
        export_module.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer()
    )
    monkeypatch.setattr(
        export_module.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: FakeModel()
    )
    monkeypatch.setattr(PeftModel, "from_pretrained", FakePeftModel.from_pretrained)
    monkeypatch.setattr(safetensors_torch, "load_file", lambda *_a, **_k: {"lora": torch.ones(2)})
    monkeypatch.setattr(export_module, "_free_memory", lambda _device: None)

    report = validate_export(config, directory, kind="adapter")

    assert report["reload_verified"] is True
    assert ("adapter",) in calls
    assert ("forward",) in calls


def test_adapter_export_can_skip_model_reload_when_configured(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from safetensors import torch as safetensors_torch
    import loraflow.export as export_module

    config = make_config(tmp_path)
    config.export.verify_reload = False
    directory = write_minimal_adapter(tmp_path / "adapter")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self):
            return 10

    monkeypatch.setattr(
        export_module.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer()
    )
    monkeypatch.setattr(safetensors_torch, "load_file", lambda *_a, **_k: {"lora": torch.ones(2)})

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Base model should not load when verify_reload is disabled")

    monkeypatch.setattr(export_module.AutoModelForCausalLM, "from_pretrained", unexpected_load)

    report = validate_export(config, directory, kind="adapter")

    assert report["reload_verified"] is False
