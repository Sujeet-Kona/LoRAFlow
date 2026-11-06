"""Adapter export, optional merge-and-save, and artifact validation.

An export is only reported as successful once the saved directory reloads as a
working model, so a partially written or incompatible artifact fails loudly
instead of being described as ready.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraflow.config import AppConfig
from loraflow.runtime import (
    cpu_threads,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    timestamp,
    write_json,
)
from loraflow.training import ADAPTER_DIRNAME, METADATA_FILENAME, _dtype_kwarg_name, load_tokenizer

LOGGER = logging.getLogger("loraflow.export")

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
MERGED_FILES = ("config.json",)


class ExportValidationError(Exception):
    """Raised when an exported directory cannot be reloaded as expected."""


def _resolve_adapter_dir(config: AppConfig, adapter_dir: str | Path | None) -> Path:
    path = Path(adapter_dir) if adapter_dir else Path(config.training.output_dir) / ADAPTER_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(f"No adapter directory at {path}. Run 'loraflow train' first.")
    return path


def _copy_if_present(source: Path, target: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for name in names:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
            copied.append(name)
    return copied


def export_adapter(
    config: AppConfig,
    *,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Copy adapter weights plus tokenizer files into a clean export folder."""

    source = _resolve_adapter_dir(config, adapter_dir)
    target = Path(output_dir) if output_dir else Path(config.export.output_dir) / "adapter"
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    copied = _copy_if_present(source, target, ADAPTER_FILES)
    copied += _copy_if_present(source, target, TOKENIZER_FILES)
    # Special tokens map and vocab files vary by tokenizer family; copy the rest
    # of the small metadata files rather than guessing names per model.
    for extra in ("special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        if (source / extra).is_file() and extra not in copied:
            shutil.copy2(source / extra, target / extra)
            copied.append(extra)
    if (source / METADATA_FILENAME).is_file():
        shutil.copy2(source / METADATA_FILENAME, target / METADATA_FILENAME)

    report = validate_export(config, target, kind="adapter")
    report["copied_files"] = sorted(copied)
    report["source_adapter_dir"] = str(source)
    write_json(target / "export_manifest.json", report)
    LOGGER.info("exported adapter to %s (%d files)", target, len(copied))

    if tracker is not None:
        tracker.log_export(report)
    return report


def merge_and_export(
    config: AppConfig,
    *,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_prompt: str | None = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Fold the LoRA weights into the base model and save a standalone model.

    Merging is exact arithmetic on the weights, so the merged model should
    reproduce the adapter's logits. That equivalence is checked here; a mismatch
    raises rather than shipping a silently different model.
    """

    from peft import PeftModel

    torch.set_num_threads(cpu_threads())
    device = resolve_device(config.training.device)
    dtype = resolve_dtype(device, config.training.dtype)
    set_global_seed(config.training.seed, device)

    source = _resolve_adapter_dir(config, adapter_dir)
    target = Path(output_dir) if output_dir else Path(config.export.merged_dir)
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    base_model_name = _base_model_name(config, source)
    LOGGER.info("merging %s + %s -> %s", base_model_name, source, target)

    tokenizer = load_tokenizer(source)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, **{_dtype_kwarg_name(): dtype}
    )
    peft_model = PeftModel.from_pretrained(base_model, str(source))
    peft_model.eval()

    probe_prompt = sample_prompt if sample_prompt is not None else _probe_prompt(config)
    reference_ids = _token_logprobs(peft_model, tokenizer, probe_prompt)

    merged = peft_model.merge_and_unload()
    merged.eval()
    if config.export.save_dtype != "same":
        merged = merged.to(_save_dtype(config.export.save_dtype))

    merged.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    del merged, peft_model

    reloaded = AutoModelForCausalLM.from_pretrained(
        target, **{_dtype_kwarg_name(): _save_dtype(config.export.save_dtype)}
    ).to(device)
    reloaded.eval()
    merged_ids = _token_logprobs(reloaded, tokenizer, probe_prompt)
    max_logit_delta = _max_abs_delta(reference_ids, merged_ids)
    del reloaded
    _free_memory(device)

    tolerance = 0.05 if config.export.save_dtype in ("same", "fp32") else 0.25
    if max_logit_delta > tolerance:
        raise ExportValidationError(
            f"Merged model diverges from the adapter by {max_logit_delta:.4f} "
            f"(tolerance {tolerance}). Export rejected."
        )

    report = validate_export(config, target, kind="merged")
    report["merge"] = {
        "base_model": base_model_name,
        "adapter_dir": str(source),
        "probe_prompt_chars": len(probe_prompt),
        "max_logit_delta": round(max_logit_delta, 6),
        "tolerance": tolerance,
        "save_dtype": config.export.save_dtype,
    }
    write_json(target / "export_manifest.json", report)
    LOGGER.info("merged export verified at %s (delta %.4f)", target, max_logit_delta)

    if tracker is not None:
        tracker.log_export(report)
    return report


def validate_export(config: AppConfig, path: str | Path, *, kind: str) -> dict[str, Any]:
    """Check that a directory reloads as the artifact it claims to be."""

    target = Path(path)
    if not target.is_dir():
        raise ExportValidationError(f"Export path is not a directory: {target}")

    required = {
        "adapter": (*ADAPTER_FILES, *TOKENIZER_FILES),
        "merged": (*MERGED_FILES, *TOKENIZER_FILES),
    }[kind]
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise ExportValidationError(f"{kind} export at {target} is missing {missing}")

    sizes = {item.name: item.stat().st_size for item in sorted(target.iterdir()) if item.is_file()}
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        raise ExportValidationError(f"Export contains zero-byte files: {empty}")

    tokenizer = AutoTokenizer.from_pretrained(str(target))
    if kind == "adapter":
        from safetensors.torch import load_file

        weights = load_file(str(target / "adapter_model.safetensors"))
        if not weights:
            raise ExportValidationError("Adapter safetensors file contains no tensors")
        parameter_count = sum(tensor.numel() for tensor in weights.values())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(target), **{_dtype_kwarg_name(): torch.float32}
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        del model

    return {
        "validated_at_utc": timestamp(),
        "kind": kind,
        "path": str(target),
        "files": sizes,
        "total_parameter_count": parameter_count,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "transformers_version": transformers.__version__,
    }


def _base_model_name(config: AppConfig, adapter_dir: Path) -> str:
    """Prefer the model the adapter was actually trained against."""

    for candidate in (
        adapter_dir / METADATA_FILENAME,
        Path(config.training.output_dir) / METADATA_FILENAME,
    ):
        if candidate.is_file():
            from loraflow.runtime import read_json

            name = read_json(candidate).get("base_model")
            if name:
                return str(name)
    return config.training.base_model


def _probe_prompt(config: AppConfig) -> str:
    """Use a real held-out prompt so the equivalence check is on-distribution."""

    from loraflow.training import prepare_splits

    fallback = "Explain what a LoRA adapter is in one sentence."
    try:
        _, held_out = prepare_splits(config)
    except (FileNotFoundError, ValueError) as exc:
        # An export-only checkout may not carry the dataset files.
        LOGGER.warning("Dataset unavailable for merge probe (%s); using static prompt", exc)
        return fallback
    return held_out[0].prompt if held_out else fallback


def _token_logprobs(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**encoded).logits[0, -1].float().cpu()


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ExportValidationError(
            f"Logit shape changed during merge: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    return float((a - b).abs().max())


def _save_dtype(name: str) -> Any:
    if name == "same":
        return torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _free_memory(device: str) -> None:
    """Release cached accelerator memory only when that backend is active."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
