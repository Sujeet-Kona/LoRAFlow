"""Tests for the training stage's own logic.

These cover LoRAFlow's mapping code -- splits, dataset shape, PEFT/TRL config
translation, checkpoint selection -- and deliberately avoid loading weights.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")

from peft import LoraConfig  # noqa: E402
from trl import SFTConfig  # noqa: E402

from loraflow.config import (  # noqa: E402
    AppConfig,
    DataConfig,
    TrainingConfig,  # noqa: E402
)
from loraflow.config import LoRAConfig as LoRASection
from loraflow.data import format_record  # noqa: E402
from loraflow.training import (  # noqa: E402
    _best_checkpoint,
    _dtype_kwarg_name,
    build_lora_config,
    build_sft_config,
    count_adapter_parameters,
    prepare_splits,
    to_hf_dataset,
    validate_examples_fit,
)


def make_config(tmp_path: Path, **data_overrides) -> AppConfig:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps({"instruction": "Label", "input": f"note {i}", "output": f"out {i}"}) + "\n"
            for i in range(20)
        ),
        encoding="utf-8",
    )
    data = DataConfig(train_file=train, **data_overrides)
    return AppConfig(
        data=data,
        lora=LoRASection(rank=4, alpha=8, target_modules=["q_proj", "v_proj"]),
        training=TrainingConfig(base_model="fake/model", output_dir=tmp_path / "run"),
    )


def examples(n: int) -> list:
    return [
        format_record(
            {"instruction": "Label", "input": f"note {i}", "output": f"out {i}"},
            line_number=i,
        )
        for i in range(n)
    ]


def test_prepare_splits_carves_validation_from_train(tmp_path: Path) -> None:
    config = make_config(tmp_path, validation_size=0.25)

    train, validation = prepare_splits(config)

    assert len(train) == 15
    assert len(validation) == 5
    assert {e.text for e in train}.isdisjoint({e.text for e in validation})


def test_prepare_splits_is_reproducible(tmp_path: Path) -> None:
    config = make_config(tmp_path, validation_size=0.2)

    first = prepare_splits(config)
    second = prepare_splits(config)

    assert [e.text for e in first[1]] == [e.text for e in second[1]]


def test_prepare_splits_prefers_validation_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    validation = tmp_path / "val.jsonl"
    validation.write_text(
        '{"instruction": "Label", "input": "held out", "output": "held"}\n', encoding="utf-8"
    )
    config.data.validation_file = validation

    train, held_out = prepare_splits(config)

    assert len(train) == 20
    assert len(held_out) == 1
    assert held_out[0].completion == "held"


def test_prepare_splits_respects_max_train_examples(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.data.max_train_examples = 10

    train, validation = prepare_splits(config)

    assert len(train) + len(validation) == 10
    assert len(validation) == 1


def test_prepare_splits_reports_an_impossible_holdout(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.data.max_train_examples = 4
    config.data.validation_size = 0.1

    # 10% of 4 rounds down to zero, so the split refuses rather than silently
    # evaluating on nothing.
    with pytest.raises(ValueError, match="cannot reserve a validation split"):
        prepare_splits(config)


def test_to_hf_dataset_uses_prompt_completion_columns(tmp_path: Path) -> None:
    dataset = to_hf_dataset(examples(3))

    assert dataset.column_names == ["prompt", "completion"]
    assert dataset[0]["completion"] == "out 0"
    assert dataset[0]["prompt"].endswith("### Response:\n")


def test_build_lora_config_maps_every_field() -> None:
    config = make_config(Path("/tmp"))
    lora = build_lora_config(config)

    assert isinstance(lora, LoraConfig)
    assert lora.r == 4
    assert lora.lora_alpha == 8
    assert set(lora.target_modules) == {"q_proj", "v_proj"}
    assert str(lora.task_type) in {"TaskType.CAUSAL_LM", "CAUSAL_LM"}


def test_build_sft_config_translates_training_knobs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.training.learning_rate = 1e-3
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 3
    config.training.warmup_steps = 5
    config.data.max_seq_length = 128

    args = build_sft_config(
        config,
        tmp_path / "run",
        device="cpu",
        dtype=pytest.importorskip("torch").float32,
        has_validation=True,
    )

    assert isinstance(args, SFTConfig)
    assert args.learning_rate == 1e-3
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 3
    assert args.warmup_steps == 5
    assert args.max_length == 128
    assert args.model_init_kwargs is None
    assert args.completion_only_loss is True
    assert args.remove_unused_columns is False
    assert args.eval_strategy == "steps"
    assert args.save_strategy == "steps"
    assert args.save_steps == config.training.save_steps
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False
    assert args.report_to == []


def test_build_sft_config_disables_evaluation_without_holdout(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    args = build_sft_config(
        config,
        tmp_path / "run",
        device="cpu",
        dtype=pytest.importorskip("torch").float32,
        has_validation=False,
    )

    assert args.eval_strategy == "no"
    assert args.eval_steps is None


def test_build_sft_config_keeps_mps_out_of_cuda_precision_flags(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    torch = pytest.importorskip("torch")

    args = build_sft_config(
        config, tmp_path / "run", device="mps", dtype=torch.float32, has_validation=True
    )

    assert args.bf16 is False and args.fp16 is False


def test_dtype_kwarg_name_matches_installed_transformers() -> None:
    import transformers

    assert _dtype_kwarg_name() == (
        "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    )


def test_best_checkpoint_picks_lowest_eval_loss(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-20").mkdir()
    (tmp_path / "checkpoint-40").mkdir()
    log_history = [
        {"step": 20, "eval_loss": 0.4},
        {"step": 40, "eval_loss": 0.25},
        {"step": 41, "loss": 0.1},
    ]

    name, value = _best_checkpoint(tmp_path, log_history)

    assert name == "checkpoint-40"
    assert value == 0.25


def test_best_checkpoint_returns_none_without_evaluations(tmp_path: Path) -> None:
    assert _best_checkpoint(tmp_path, [{"step": 5, "loss": 1.0}]) == (None, None)


def test_best_checkpoint_ignores_deleted_directory(tmp_path: Path) -> None:
    name, value = _best_checkpoint(tmp_path, [{"step": 10, "eval_loss": 0.5}])

    assert name is None
    assert value == 0.5


def test_count_adapter_parameters_sees_only_lora_tensors() -> None:
    torch = pytest.importorskip("torch")

    class Fake:
        def parameters(self):
            frozen = torch.nn.Parameter(torch.zeros(3, 3), requires_grad=False)
            adapter = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=True)
            return iter([frozen, adapter])

    assert count_adapter_parameters(Fake()) == (4, 13)


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, texts, *, add_special_tokens=False):
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


def test_training_rejects_examples_that_would_be_truncated() -> None:
    example = format_record(
        {"instruction": "Label", "input": "note", "output": "urgent response"},
        line_number=12,
    )

    with pytest.raises(ValueError, match="source line 12.*max_seq_length=3"):
        validate_examples_fit(FakeTokenizer(), [example], max_length=3)


def test_training_accepts_examples_at_the_eos_inclusive_limit() -> None:
    example = format_record(
        {"instruction": "Label", "input": "note", "output": "urgent response"},
        line_number=1,
    )

    validate_examples_fit(FakeTokenizer(), [example], max_length=11)
