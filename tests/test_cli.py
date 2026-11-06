"""CLI surface tests: argument wiring and error exits, no model loading."""

import json
from pathlib import Path

import pytest

from loraflow.cli import build_parser, main


def write_config(tmp_path: Path, **overrides) -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps({"instruction": "Label", "input": f"n{i}", "output": f"o{i}"}) + "\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    config = {
        "data": {"train_file": str(train), "max_seq_length": 64},
        "training": {"base_model": "fake/model", "output_dir": str(tmp_path / "run")},
        "tracking": {"enabled": False},
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_parser_exposes_every_stage() -> None:
    parser = build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)  # noqa: SLF001

    assert {
        "data-check",
        "train",
        "evaluate",
        "export",
        "serve",
        "mlflow-ui",
    } <= commands


def test_config_defaults_to_the_documented_path() -> None:
    args = build_parser().parse_args(["data-check"])

    assert args.config == Path("configs/default.yaml")


def test_config_flag_is_overridable(tmp_path: Path) -> None:
    path = write_config(tmp_path)

    args = build_parser().parse_args(["train", "--config", str(path), "--no-track"])

    assert args.config == path
    assert args.no_track is True


def test_train_flags_reach_namespace() -> None:
    args = build_parser().parse_args(
        ["train", "--device", "cpu", "--epochs", "2", "--max-steps", "3", "--resume", "ckpt"]
    )

    assert (args.device, args.epochs, args.max_steps, args.resume) == ("cpu", 2, 3, "ckpt")


def test_missing_config_exits_with_two(tmp_path: Path) -> None:
    assert main(["data-check", "--config", str(tmp_path / "nope.yaml")]) == 2


def test_invalid_config_value_exits_with_two(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  base_model: x\n", encoding="utf-8")

    # Missing the required data section, which pydantic reports as a ValueError.
    assert main(["data-check", "--config", str(path)]) == 2


def test_malformed_dataset_exits_with_two(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text('{"text": "no instruction field"}\n', encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text(
        json.dumps({"data": {"train_file": str(train)}, "training": {"base_model": "m"}}),
        encoding="utf-8",
    )

    # YAML accepts JSON, and the dataset error surfaces as exit code 2.
    assert main(["data-check", "--config", str(config)]) == 2


def test_data_check_prints_split_statistics(tmp_path: Path, capsys) -> None:
    pytest.importorskip("trl")
    path = write_config(tmp_path)

    assert main(["data-check", "--config", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["train"]["examples"] + payload["validation"]["examples"] == 12
    assert payload["validation_source"] == "split_from_train"
    assert payload["validation"]["examples"] == 1
    assert payload["sample_completion"] in {
        "o0",
        "o1",
        "o2",
        "o3",
        "o4",
        "o5",
        "o6",
        "o7",
        "o8",
        "o9",
        "o10",
        "o11",
    }


def test_export_reports_missing_adapter(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    path = write_config(tmp_path)

    assert main(["export", "--config", str(path)]) == 2


def test_mlflow_ui_uses_the_backend_store_option(tmp_path: Path, monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda argv: captured.append(argv) or 0)
    path = write_config(tmp_path, tracking={"tracking_uri": "sqlite:///other.db"})

    assert main(["mlflow-ui", "--config", str(path), "--port", "5050"]) == 0

    argv = captured[0]
    # mlflow ui rejects --tracking-uri, so the URI has to travel as --backend-store-uri.
    assert "--tracking-uri" not in argv
    assert argv[argv.index("--backend-store-uri") + 1] == "sqlite:///other.db"
    assert argv[argv.index("--port") + 1] == "5050"
