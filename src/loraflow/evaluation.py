"""Held-out evaluation: baseline base model versus the fine-tuned adapter.

Both variants are scored on exactly the same examples with the same prompt
template and the same greedy decoding settings. Every number in the report is
computed in this module from this run; nothing is quoted from an external
benchmark, and generation is skipped entirely when the config asks for zero
generated examples.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM

from loraflow.config import AppConfig, config_sha256
from loraflow.data import Example, dataset_report, file_identity, load_examples
from loraflow.metrics import field_scores, format_delta, perplexity_from_loss, score_generations
from loraflow.runtime import (
    cpu_threads,
    read_json,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    timestamp,
    write_json,
)
from loraflow.training import (
    ADAPTER_DIRNAME,
    METADATA_FILENAME,
    _dtype_kwarg_name,
    load_tokenizer,
    prepare_splits,
)

LOGGER = logging.getLogger("loraflow.evaluation")

IGNORE_INDEX = -100
# Metrics compared in the "delta" block: name -> higher_is_better.
_COMPARED = (
    ("mean_answer_loss", False),
    ("perplexity", False),
    ("answer_token_accuracy", True),
    ("exact_match", True),
    ("prefix_match", True),
)


def _build_scoring_row(
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    max_length: int,
    source_line: int,
) -> tuple[list[int], list[int]]:
    """Build an untruncated scoring row or fail with the offending record."""

    total = len(prompt_ids) + len(completion_ids)
    if total > max_length:
        raise ValueError(
            f"Example on source line {source_line} needs {total} tokens "
            f"(prompt={len(prompt_ids)}, completion={len(completion_ids)}), "
            f"exceeding max_seq_length={max_length}; increase the limit or shorten the record."
        )
    input_ids = prompt_ids + completion_ids
    return input_ids, _answer_labels(prompt_ids, completion_ids, input_ids)


def _answer_labels(
    prompt_ids: list[int], completion_ids: list[int], input_ids: list[int]
) -> list[int]:
    """Mask every position except the answer span.

    This matches the training objective, so the reported loss measures the
    part of the sequence the adapter was actually optimised on.
    """

    labels = [IGNORE_INDEX] * len(input_ids)
    for offset in range(len(completion_ids)):
        index = len(prompt_ids) + offset
        if index < len(input_ids):
            labels[index] = input_ids[index]
    return labels


@torch.no_grad()
def score_teacher_forcing(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Answer-token loss, perplexity and next-token accuracy under teacher forcing."""

    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    total_loss = 0.0
    examples_scored = 0
    correct = 0
    scored_tokens = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = tokenizer([e.prompt for e in batch], add_special_tokens=False)["input_ids"]
        completions = tokenizer([e.completion for e in batch], add_special_tokens=False)[
            "input_ids"
        ]

        rows = []
        for example, prompt, completion in zip(batch, prompts, completions, strict=True):
            rows.append(
                _build_scoring_row(
                    prompt,
                    completion,
                    max_length=max_length,
                    source_line=example.source_line,
                )
            )

        width = max(len(input_ids) for input_ids, _ in rows)
        input_batch, mask_batch, label_batch = [], [], []
        for input_ids, labels in rows:
            pad = width - len(input_ids)
            input_batch.append(input_ids + [pad_id] * pad)
            mask_batch.append([1] * len(input_ids) + [0] * pad)
            label_batch.append(labels + [IGNORE_INDEX] * pad)

        device = next(model.parameters()).device
        outputs = model(
            input_ids=torch.tensor(input_batch, device=device),
            attention_mask=torch.tensor(mask_batch, device=device),
            labels=torch.tensor(label_batch, device=device),
        )

        logits = outputs.logits.float()
        # logits[:, i] predicts token i+1, so shift both by one position.
        labels_tensor = torch.tensor(label_batch, device=device)[:, 1:]
        shift_labels = labels_tensor.reshape(-1)
        per_position = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            shift_labels,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(labels_tensor.shape)

        valid = labels_tensor != IGNORE_INDEX
        for row in range(valid.size(0)):
            mask = valid[row]
            count = int(mask.sum())
            if count == 0:
                continue
            total_loss += float(per_position[row][mask].sum())
            examples_scored += 1
            row_logits = logits[row, :-1, :][mask]
            correct += int((row_logits.argmax(dim=-1) == labels_tensor[row][mask]).sum())
            scored_tokens += count

    if not scored_tokens:
        raise ValueError(
            "Teacher forcing scored no answer tokens; the prompt template must "
            "leave a completion span after tokenisation."
        )

    # Aggregate token NLL globally; averaging per-example means would give
    # short completions the same weight as long completions.
    mean_loss = total_loss / scored_tokens
    return {
        "examples_scored": examples_scored,
        "answer_tokens": scored_tokens,
        "mean_answer_loss": round(mean_loss, 6),
        "perplexity": round(perplexity_from_loss(mean_loss), 4),
        "answer_token_accuracy": round(correct / max(scored_tokens, 1), 4),
    }


@torch.no_grad()
def generate_answers(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    *,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Complete each prompt and return only the newly generated span."""

    model.eval()
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # Keeps every prompt adjacent to its new tokens.
    generations: list[str] = []
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = tokenizer(
                [e.prompt for e in batch],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(next(model.parameters()).device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            if do_sample:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            outputs = model.generate(**encoded, **kwargs)
            new_tokens = outputs[:, encoded["input_ids"].size(1) :]
            for row in range(new_tokens.size(0)):
                text = tokenizer.decode(new_tokens[row], skip_special_tokens=True)
                # Cut at the next section header in case the model continues
                # past its own answer into a fabricated "### Input:" block.
                generations.append(text.split("###")[0].strip())
    finally:
        tokenizer.padding_side = previous_side
    return generations


def load_adapter_model(
    base_model_name: str, adapter_dir: Path, *, device: str, dtype: torch.dtype
) -> tuple[Any, Any]:
    """Load the base model with a saved LoRA adapter attached."""

    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **{_dtype_kwarg_name(): dtype})
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.config.use_cache = True
    return model, load_tokenizer(adapter_dir)


def _score_variant(
    model: Any,
    tokenizer: Any,
    held_out: list[Example],
    gen_examples: list[Example],
    config: AppConfig,
) -> dict[str, Any]:
    block = score_teacher_forcing(
        model,
        tokenizer,
        held_out,
        max_length=config.data.max_seq_length,
        batch_size=config.evaluation.batch_size,
    )
    model.eval()
    model.config.use_cache = True
    if gen_examples:
        predictions = generate_answers(
            model,
            tokenizer,
            gen_examples,
            max_new_tokens=config.evaluation.max_new_tokens,
            batch_size=config.evaluation.batch_size,
            do_sample=config.evaluation.do_sample,
            temperature=config.evaluation.temperature,
            top_p=config.evaluation.top_p,
        )
        golds = [example.completion for example in gen_examples]
        block.update(score_generations(predictions, golds).to_dict())
        block["per_field"] = field_scores(predictions, golds)
        block["predictions"] = predictions
    else:
        block["generation_examples"] = 0
    return block


def evaluate_config(
    config: AppConfig,
    *,
    adapter_dir: str | Path | None = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Score the base model and the adapter on identical held-out examples."""

    torch.set_num_threads(cpu_threads())
    device = resolve_device(config.training.device)
    dtype = resolve_dtype(device, config.training.dtype)
    set_global_seed(config.training.seed, device)

    adapter_path = (
        Path(adapter_dir) if adapter_dir else Path(config.training.output_dir) / ADAPTER_DIRNAME
    )
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"No adapter directory at {adapter_path}. Run 'loraflow train' first."
        )

    if config.data.test_file is not None:
        held_out = load_examples(
            config.data.test_file,
            instruction_field=config.data.instruction_field,
            input_field=config.data.input_field,
            output_field=config.data.output_field,
            template=config.data.prompt_template,
        )
        held_out_source = "test_file"
    else:
        _, held_out = prepare_splits(config)
        held_out_source = (
            "validation_file_fallback"
            if config.data.validation_file
            else "split_from_train_fallback"
        )
    if config.evaluation.max_examples:
        held_out = held_out[: config.evaluation.max_examples]
    if not held_out:
        raise ValueError("Evaluation needs at least one held-out example")
    if config.evaluation.generation_examples < 0:
        raise ValueError("evaluation.generation_examples cannot be negative")
    gen_examples = held_out[: config.evaluation.generation_examples]

    LOGGER.info(
        "scoring %d held-out examples, generating for %d of them",
        len(held_out),
        len(gen_examples),
    )

    metadata_path = Path(config.training.output_dir) / METADATA_FILENAME
    run_metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    base_model_name = run_metadata.get("base_model", config.training.base_model)

    tokenizer = load_tokenizer(base_model_name)

    LOGGER.info("baseline: %s", base_model_name)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, **{_dtype_kwarg_name(): dtype}
    ).to(device)
    baseline_model.config.use_cache = False
    baseline = _score_variant(baseline_model, tokenizer, held_out, gen_examples, config)
    del baseline_model
    _free_memory(device)

    LOGGER.info("fine-tuned: adapter %s", adapter_path)
    fine_tuned_model, adapter_tokenizer = load_adapter_model(
        base_model_name, adapter_path, device=device, dtype=dtype
    )
    fine_tuned_model.config.use_cache = False
    fine_tuned = _score_variant(fine_tuned_model, adapter_tokenizer, held_out, gen_examples, config)
    del fine_tuned_model
    _free_memory(device)

    delta = {
        key: format_delta(baseline[key], fine_tuned[key])
        for key, _ in _COMPARED
        if key in baseline and key in fine_tuned
    }

    report: dict[str, Any] = {
        "measured_at_utc": timestamp(),
        "config_sha256": config_sha256(config),
        "held_out_dataset": file_identity(
            config.data.test_file
            if config.data.test_file is not None
            else config.data.validation_file or config.data.train_file
        ),
        "base_model": base_model_name,
        "adapter_dir": str(adapter_path),
        "environment": {
            "platform": f"{platform.system()}/{platform.machine()}",
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "trl_version": _trl_version(),
        },
        "reproducibility": {
            "seed": config.training.seed,
            "decoding": "greedy" if not config.evaluation.do_sample else "sampling",
            "max_new_tokens": config.evaluation.max_new_tokens,
            "generation_batch_size": config.evaluation.batch_size,
            "note": (
                "Both blocks score the same held-out examples with the same "
                "prompt template. All figures were measured by this run; no "
                "external benchmark number is reproduced here."
            ),
        },
        "held_out": dataset_report(held_out),
        "held_out_source": held_out_source,
        "generation_examples": len(gen_examples),
        "baseline": baseline,
        "fine_tuned": fine_tuned,
        "delta": delta,
        "field_accuracy": {
            name: {
                "baseline": baseline.get("per_field", {}).get(name, {}).get("accuracy"),
                "fine_tuned": fine_tuned.get("per_field", {}).get(name, {}).get("accuracy"),
            }
            for name in sorted(
                set(baseline.get("per_field", {})) | set(fine_tuned.get("per_field", {}))
            )
        },
        "examples": [
            {
                "input": _note_text(example),
                "gold": example.completion,
                "baseline": base_pred,
                "fine_tuned": tuned_pred,
            }
            for example, base_pred, tuned_pred in zip(
                gen_examples,
                baseline.get("predictions", []),
                fine_tuned.get("predictions", []),
                strict=True,
            )
        ],
    }

    write_json(config.evaluation.output_file, report)
    LOGGER.info("wrote evaluation report to %s", config.evaluation.output_file)

    if tracker is not None:
        try:
            tracker.log_evaluation(report)
        finally:
            tracker.finish()
    return report


def _note_text(example: Example) -> str:
    """The raw note from a formatted prompt, for readable report samples."""

    return example.prompt.split("### Input:")[-1].split("### Response:")[0].strip()


def _trl_version() -> str:
    import trl

    return trl.__version__


def _free_memory(device: str) -> None:
    """Release cached accelerator memory only when that backend is active."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
