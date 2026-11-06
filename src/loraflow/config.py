"""Typed configuration loading for LoRAFlow."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

DeviceName = Literal["auto", "cpu", "mps", "cuda"]
DtypeName = Literal["auto", "fp32", "bf16", "fp16"]


class DataConfig(BaseModel):
    """Input dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    train_file: Path
    validation_file: Path | None = None
    test_file: Path | None = None
    max_seq_length: PositiveInt = 1024

    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    prompt_template: str = "### Instruction:\n{instruction}\n{input_block}### Response:\n"

    # Used when validation_file is absent: the held-out slice is carved from
    # train_file with this fraction (or count) and the training seed.
    validation_size: float = Field(
        default=0.1,
        gt=0.0,
        description="Fraction in (0, 1) or an absolute example count when >= 1.",
    )
    max_train_examples: PositiveInt | None = None

    @field_validator("validation_size")
    @classmethod
    def absolute_validation_size_must_be_an_integer(cls, value: float) -> float:
        if value >= 1 and not value.is_integer():
            raise ValueError("validation_size must be a whole example count when >= 1")
        return value


class LoRAConfig(BaseModel):
    """Low-rank adapter configuration."""

    model_config = ConfigDict(extra="forbid")

    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"], min_length=1
    )


class TrainingConfig(BaseModel):
    """Training run configuration."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    output_dir: Path = Path("artifacts")
    num_train_epochs: PositiveInt = 1
    learning_rate: float = Field(default=2e-4, gt=0.0)
    per_device_train_batch_size: PositiveInt = 1
    gradient_accumulation_steps: PositiveInt = 4
    seed: int = 42

    device: DeviceName = "auto"
    dtype: DtypeName = "auto"
    # TRL's SFTConfig exposes warmup_steps (not warmup_ratio), so this knob is
    # an absolute step count to keep the mapping to the library honest.
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_scheduler: str = "linear"
    max_steps: int = Field(default=-1, ge=-1)
    logging_steps: PositiveInt = 10
    save_steps: PositiveInt = 500
    eval_steps: PositiveInt = 25
    eval_batch_size: PositiveInt = 4
    gradient_checkpointing: bool = False
    # Optional cap to keep a smoke run short without editing epochs.
    run_name: str | None = None


class EvaluationConfig(BaseModel):
    """Held-out comparison between the base model and the adapter."""

    model_config = ConfigDict(extra="forbid")

    output_file: Path = Path("artifacts/evaluation_report.json")
    # Held-out examples scored with teacher forcing (loss, perplexity,
    # token accuracy). 0 means "score every example".
    max_examples: int = Field(default=0, ge=0)
    # Number of examples also generated from, used for the exact-match metric.
    generation_examples: int = Field(default=20, ge=0)
    max_new_tokens: PositiveInt = 48
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    batch_size: PositiveInt = 4


class TrackingConfig(BaseModel):
    """MLflow experiment tracking."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # MLflow 3.x treats the plain filesystem store as deprecated, so the local
    # default is a SQLite store; artifacts still land in artifacts_dir.
    tracking_uri: str = "sqlite:///mlflow.db"
    artifacts_dir: str = "file://./mlartifacts"
    experiment_name: str = "loraflow"
    run_name: str | None = None
    log_artifacts: bool = True


class ExportConfig(BaseModel):
    """Adapter export and optional merge-and-save workflow."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("artifacts/exported")
    merge_and_save: bool = False
    merged_dir: Path = Path("artifacts/merged")
    save_dtype: Literal["same", "fp32", "bf16", "fp16"] = "same"
    verify_reload: bool = True


class ServingConfig(BaseModel):
    """FastAPI inference service."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    device: DeviceName = "auto"
    max_new_tokens: PositiveInt = 64
    max_input_chars: PositiveInt = 4000
    max_concurrent_requests: PositiveInt = 2
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    return AppConfig.model_validate(raw)


def config_sha256(config: AppConfig) -> str:
    """Stable identity of a resolved configuration.

    Serialised through ``model_dump`` rather than the file, so defaults that were
    never written down still hash identically and two runs are comparable even if
    one config spelled out the default and the other inherited it.
    """

    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
