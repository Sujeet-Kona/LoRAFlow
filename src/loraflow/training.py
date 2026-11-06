"""LoRA supervised fine-tuning on top of Transformers, PEFT and TRL.

Importing this module requires the ``train`` extra.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from loraflow.config import AppConfig, config_sha256
from loraflow.data import (
    Example,
    dataset_provenance,
    dataset_report,
    deduplicate,
    file_identity,
    load_examples,
    split_examples,
)
from loraflow.runtime import (
    cpu_threads,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    timestamp,
    write_json,
)

LOGGER = logging.getLogger("loraflow.training")

ADAPTER_DIRNAME = "adapter"
METADATA_FILENAME = "training_metadata.json"


@dataclass(frozen=True)
class TrainingResult:
    """On-disk outputs of a completed run."""

    output_dir: Path
    adapter_dir: Path
    metadata_path: Path
    best_metric: float | None
    best_checkpoint: str | None
    steps: int
    device: str
    dtype: str


def prepare_splits(config: AppConfig) -> tuple[list[Example], list[Example]]:
    """Resolve the training set and the held-out split.

    With ``data.validation_file`` set it is used verbatim; otherwise the
    held-out slice is carved from the training file using the configured
    fraction and seed, which keeps the split identical across reruns.
    """

    train_examples = deduplicate(
        load_examples(
            config.data.train_file,
            instruction_field=config.data.instruction_field,
            input_field=config.data.input_field,
            output_field=config.data.output_field,
            template=config.data.prompt_template,
            max_records=config.data.max_train_examples,
        )
    )

    if config.data.validation_file is not None:
        validation_examples = load_examples(
            config.data.validation_file,
            instruction_field=config.data.instruction_field,
            input_field=config.data.input_field,
            output_field=config.data.output_field,
            template=config.data.prompt_template,
        )
        return train_examples, validation_examples

    return split_examples(
        train_examples,
        validation_size=config.data.validation_size,
        seed=config.training.seed,
    )


def to_hf_dataset(examples: list[Example]) -> Dataset:
    """Build TRL's prompt-completion layout.

    TRL tokenises prompt and completion separately and, with
    ``completion_only_loss=True``, masks the prompt positions, so the loss is
    computed over answers only. It also appends the EOS token to each
    completion, which is what lets the served model stop by itself.
    """

    return Dataset.from_dict(
        {
            "prompt": [example.prompt for example in examples],
            "completion": [example.completion for example in examples],
        }
    )


def build_lora_config(config: AppConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def _dtype_kwarg_name() -> str:
    """Transformers 5 renamed ``torch_dtype`` to ``dtype``."""

    return "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def load_tokenizer(model_name: str, *, trust_remote_code: bool = False) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # Unusual for modern decoder models, but padded batches need a pad id.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def validate_examples_fit(tokenizer: Any, examples: list[Example], *, max_length: int) -> None:
    """Reject examples that TRL would silently truncate during SFT preprocessing."""

    prompts = tokenizer([example.prompt for example in examples], add_special_tokens=False)[
        "input_ids"
    ]
    completions = tokenizer([example.completion for example in examples], add_special_tokens=False)[
        "input_ids"
    ]
    eos_id = tokenizer.eos_token_id
    for example, prompt_ids, completion_ids in zip(examples, prompts, completions, strict=True):
        total = len(prompt_ids) + len(completion_ids)
        if eos_id is not None and (not completion_ids or completion_ids[-1] != eos_id):
            total += 1  # TRL appends EOS to the completion.
        if total > max_length:
            raise ValueError(
                f"Example on source line {example.source_line} needs {total} tokens "
                f"including EOS, exceeding max_seq_length={max_length}; "
                "increase the limit or shorten the record."
            )


def load_base_model(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **{_dtype_kwarg_name(): dtype},
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    # KV caching only helps generation; it blocks gradient checkpointing.
    model.config.use_cache = False
    return model


def build_sft_config(
    config: AppConfig,
    output_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
    has_validation: bool,
) -> SFTConfig:
    training = config.training
    run_name = training.run_name or config.tracking.run_name or output_dir.name
    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        seed=training.seed,
        data_seed=training.seed,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        warmup_steps=training.warmup_steps,
        weight_decay=training.weight_decay,
        lr_scheduler_type=training.lr_scheduler,
        logging_steps=training.logging_steps,
        save_strategy="steps",
        save_steps=training.save_steps,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=training.eval_steps if has_validation else None,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        save_total_limit=2,
        per_device_eval_batch_size=training.eval_batch_size,
        gradient_checkpointing=training.gradient_checkpointing,
        bf16=device == "cuda" and dtype == torch.bfloat16,
        fp16=device == "cuda" and dtype == torch.float16,
        report_to=[],
        # The prompt/completion columns are consumed by TRL's own collator.
        remove_unused_columns=False,
        completion_only_loss=True,
        packing=False,
        max_length=config.data.max_seq_length,
        model_init_kwargs={_dtype_kwarg_name(): dtype},
    )


def count_adapter_parameters(model: Any) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts of a PEFT-wrapped model."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_training(
    config: AppConfig,
    *,
    tracker: Any = None,
    resume_from_checkpoint: str | None = None,
) -> TrainingResult:
    """Fine-tune the configured base model with LoRA and save the adapter."""

    torch.set_num_threads(cpu_threads())
    device = resolve_device(config.training.device)
    dtype = resolve_dtype(device, config.training.dtype)
    set_global_seed(config.training.seed, device)
    LOGGER.info(
        "device=%s dtype=%s base_model=%s threads=%d",
        device,
        dtype,
        config.training.base_model,
        torch.get_num_threads(),
    )

    train_examples, validation_examples = prepare_splits(config)
    run_dir = Path(config.training.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.training.base_model)
    validate_examples_fit(
        tokenizer,
        train_examples + validation_examples,
        max_length=config.data.max_seq_length,
    )
    model = load_base_model(config.training.base_model, device=device, dtype=dtype)
    peft_config = build_lora_config(config)

    sft_config = build_sft_config(
        config,
        run_dir,
        device=device,
        dtype=dtype,
        has_validation=bool(validation_examples),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=to_hf_dataset(train_examples),
        eval_dataset=to_hf_dataset(validation_examples) if validation_examples else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainable, total = count_adapter_parameters(trainer.model)
    LOGGER.info(
        "trainable %s of %s params (%.3f%%) | train=%d held_out=%d",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(total, 1),
        len(train_examples),
        len(validation_examples),
    )

    history = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # transformers 5 returns (global_step, training_loss, metrics); the epoch
    # count and per-eval results live on the trainer state.
    log_history = list(trainer.state.log_history)
    train_metrics = history.metrics or {}

    adapter_dir = run_dir / ADAPTER_DIRNAME
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    best_checkpoint, best_metric = _best_checkpoint(run_dir, log_history)
    steps = int(history.global_step or 0)

    metadata: dict[str, Any] = {
        "created_at_utc": timestamp(),
        "config_sha256": config_sha256(config),
        "dataset_provenance": {
            **dataset_provenance(config.data.train_file, config.data.validation_file),
            "test": file_identity(config.data.test_file) if config.data.test_file else None,
        },
        "run_name": sft_config.run_name,
        "base_model": config.training.base_model,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": f"{platform.system()}/{platform.machine()}",
        "seed": config.training.seed,
        "adapter": {
            **config.lora.model_dump(),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": round(trainable / max(total, 1), 6),
        },
        "training": {
            **config.training.model_dump(),
            "effective_train_batch_size": (
                config.training.per_device_train_batch_size
                * config.training.gradient_accumulation_steps
            ),
        },
        "splits": {
            "train": dataset_report(train_examples),
            "validation": dataset_report(validation_examples),
            "validation_source": (
                "validation_file" if config.data.validation_file else "split_from_train"
            ),
        },
        "steps": steps,
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "train_loss": history.training_loss,
        "runtime_seconds": train_metrics.get("train_runtime"),
        "final_eval": {
            key: value for key, value in train_metrics.items() if key.startswith("eval_")
        },
        "best_checkpoint": best_checkpoint,
        "best_eval_loss": best_metric,
        "adapter_dir": str(adapter_dir),
    }
    write_json(run_dir / METADATA_FILENAME, metadata)

    if tracker is not None:
        try:
            tracker.start(run_name=sft_config.run_name)
            tracker.log_run(config, extra={"trainable_parameters": trainable})
            tracker.log_training(metadata)
            tracker.log_adapter(adapter_dir)
        finally:
            tracker.finish()

    return TrainingResult(
        output_dir=run_dir,
        adapter_dir=adapter_dir,
        metadata_path=run_dir / METADATA_FILENAME,
        best_metric=best_metric,
        best_checkpoint=best_checkpoint,
        steps=steps,
        device=device,
        dtype=str(dtype),
    )


def _best_checkpoint(
    run_dir: Path, log_history: list[dict[str, Any]]
) -> tuple[str | None, float | None]:
    """Locate the checkpoint directory with the lowest recorded eval loss."""

    entries = [entry for entry in log_history if "eval_loss" in entry and "step" in entry]
    if not entries:
        return None, None
    best = min(entries, key=lambda entry: entry["eval_loss"])
    checkpoint = run_dir / f"checkpoint-{int(best['step'])}"
    return (checkpoint.name if checkpoint.is_dir() else None), float(best["eval_loss"])
