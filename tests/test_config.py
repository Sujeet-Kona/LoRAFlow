from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from loraflow.config import AppConfig, load_config


def valid_config() -> dict:
    return {
        "data": {"train_file": "train.jsonl"},
        "training": {"base_model": "example/model"},
    }


def test_load_config_validates_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(valid_config()), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, AppConfig)
    assert config.lora.rank == 8
    assert config.training.base_model == "example/model"


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = valid_config()
    raw["training"]["unexpected"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_requires_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_data_config_accepts_absolute_validation_count() -> None:
    config = AppConfig.model_validate(
        {
            **valid_config(),
            "data": {"train_file": "train.jsonl", "validation_size": 5},
        }
    )

    assert config.data.validation_size == 5


def test_data_config_rejects_non_positive_validation_size() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {**valid_config(), "data": {"train_file": "train.jsonl", "validation_size": 0}}
        )


def test_config_rejects_fractional_absolute_validation_count() -> None:
    with pytest.raises(ValidationError, match="whole example count"):
        AppConfig.model_validate(
            {**valid_config(), "data": {"train_file": "train.jsonl", "validation_size": 2.5}}
        )


@pytest.mark.parametrize("port", [0, 65536])
def test_config_rejects_invalid_serving_port(port: int) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                **valid_config(),
                "serving": {"port": port},
            }
        )


def test_config_rejects_empty_lora_target_modules() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({**valid_config(), "lora": {"target_modules": []}})


def test_training_config_rejects_misaligned_checkpoint_intervals() -> None:
    raw = valid_config()
    raw["training"].update({"save_steps": 10, "eval_steps": 3})

    with pytest.raises(ValidationError, match="save_steps must be a multiple"):
        AppConfig.model_validate(raw)


def test_training_config_validates_cli_style_assignments() -> None:
    config = AppConfig.model_validate(valid_config())

    with pytest.raises(ValidationError):
        config.training.num_train_epochs = 0

    with pytest.raises(ValidationError):
        config.training.max_steps = -2
