"""Typed configuration loading for LoRAFlow."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DataConfig(BaseModel):
    """Input dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    train_file: Path
    validation_file: Path | None = None
    text_field: str = "text"
    max_seq_length: PositiveInt = 1024


class LoRAConfig(BaseModel):
    """Low-rank adapter configuration."""

    model_config = ConfigDict(extra="forbid")

    rank: PositiveInt = 8
    alpha: PositiveInt = 16
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


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


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig


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
