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
