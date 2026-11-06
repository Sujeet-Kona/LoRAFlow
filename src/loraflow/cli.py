"""Command line entrypoint for the LoRAFlow stages.

Argparse is used deliberately: the pipeline has a handful of commands and does
not need another framework.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from loraflow import __version__
from loraflow.config import AppConfig, load_config

LOGGER = logging.getLogger("loraflow.cli")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Progress bars and HTTP access logs drown out the pipeline stages.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


def _tracker(config: AppConfig, *, no_track: bool, tags: dict[str, str]) -> Any:
    if no_track:
        from loraflow.tracking import NoopTracker

        return NoopTracker()
    from loraflow.tracking import build_tracker

    return build_tracker(config, tags=tags)


def command_data_check(args: argparse.Namespace) -> int:
    from loraflow.data import dataset_report, deduplicate, load_examples, split_examples

    config = load_config(args.config)
    data = config.data
    train = deduplicate(
        load_examples(
            data.train_file,
            instruction_field=data.instruction_field,
            input_field=data.input_field,
            output_field=data.output_field,
            template=data.prompt_template,
            max_records=data.max_train_examples,
        )
    )
    if data.validation_file is not None:
        validation = load_examples(
            data.validation_file,
            instruction_field=data.instruction_field,
            input_field=data.input_field,
            output_field=data.output_field,
            template=data.prompt_template,
        )
    else:
        train, validation = split_examples(
            train,
            validation_size=data.validation_size,
            seed=config.training.seed,
        )
    payload = {
        "config": str(config.data.train_file),
        "train": dataset_report(train),
        "validation": dataset_report(validation),
        "validation_source": (
            "validation_file" if config.data.validation_file else "split_from_train"
        ),
        "sample_prompt": train[0].prompt,
        "sample_completion": train[0].completion,
    }
    if data.test_file is not None:
        test = load_examples(
            data.test_file,
            instruction_field=data.instruction_field,
            input_field=data.input_field,
            output_field=data.output_field,
            template=data.prompt_template,
        )
        payload["test"] = dataset_report(test)
        payload["test_source"] = "test_file"
    print(json.dumps(payload, indent=2, default=str))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from loraflow.training import run_training

    config = load_config(args.config)
    if args.base_model:
        config.training.base_model = args.base_model
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
    if args.device:
        config.training.device = args.device

    tracker = _tracker(config, no_track=args.no_track, tags={"stage": "train"})
    result = run_training(config, tracker=tracker, resume_from_checkpoint=args.resume)
    print(
        json.dumps(
            {
                "adapter_dir": str(result.adapter_dir),
                "metadata": str(result.metadata_path),
                "steps": result.steps,
                "device": result.device,
                "dtype": result.dtype,
                "best_eval_loss": result.best_metric,
                "best_checkpoint": result.best_checkpoint,
                "mlflow_run_id": tracker.run_id,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from loraflow.evaluation import evaluate_config

    config = load_config(args.config)
    if args.max_examples is not None:
        config.evaluation.max_examples = args.max_examples
    if args.generation_examples is not None:
        config.evaluation.generation_examples = args.generation_examples
    tracker = _tracker(config, no_track=args.no_track, tags={"stage": "evaluate"})
    report = evaluate_config(config, adapter_dir=args.adapter, tracker=tracker)
    print(
        json.dumps(
            {"delta": report["delta"], "report": str(config.evaluation.output_file)}, indent=2
        )
    )
    return 0


def command_export(args: argparse.Namespace) -> int:
    from loraflow.export import export_adapter, merge_and_export

    config = load_config(args.config)
    # The flag overrides the config so a one-off merge needs no file edit.
    merge = args.merged or config.export.merge_and_save
    if merge:
        report = merge_and_export(config, adapter_dir=args.adapter, output_dir=args.output)
    else:
        report = export_adapter(config, adapter_dir=args.adapter, output_dir=args.output)
    print(json.dumps(report, indent=2, default=str))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from loraflow.api import ModelService, create_app

    config = load_config(args.config)
    if args.device:
        config.serving.device = args.device
    service = ModelService(
        config,
        adapter_dir=args.adapter,
        base_model=args.base_model,
        device=args.device or None,
    )
    try:
        service.load()
    except Exception as exc:
        # Keep the server up so /health explains the failure to the operator.
        LOGGER.error("Model load failed at startup: %s", exc)
    app = create_app(config, service)
    uvicorn.run(app, host=args.host, port=args.port, log_level=config.serving.log_level.lower())
    return 0


def command_mlflow_ui(args: argparse.Namespace) -> int:
    import subprocess

    config = load_config(args.config)
    uri = os.environ.get("LORAFLOW_MLFLOW_TRACKING_URI", config.tracking.tracking_uri)
    artifact_root = os.environ.get(
        "LORAFLOW_MLFLOW_ARTIFACTS_DIR", config.tracking.artifacts_dir
    )
    print(f"tracking uri: {uri}")
    # MLflow's `ui` command takes the SQLAlchemy URI as --backend-store-uri;
    # --tracking-uri is not an option on it.
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            uri,
            "--default-artifact-root",
            artifact_root,
            "--port",
            str(args.port),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loraflow", description="LoRA fine-tuning pipeline")
    parser.add_argument("--version", action="version", version=f"loraflow {__version__}")
    parser.add_argument("--log-level", default="INFO")

    # Every subcommand takes a config, so it lives on a shared parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config (default: configs/default.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "data-check", parents=[common], help="Validate the dataset and show split statistics"
    ).set_defaults(func=command_data_check)

    train = subparsers.add_parser("train", parents=[common], help="Fine-tune with LoRA")
    train.add_argument("--base-model")
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", help="Checkpoint directory to resume from")
    train.add_argument("--no-track", action="store_true", help="Skip MLflow")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser(
        "evaluate", parents=[common], help="Compare base model and adapter"
    )
    evaluate.add_argument("--adapter", help="Adapter directory (defaults to the run's)")
    evaluate.add_argument("--max-examples", type=int)
    evaluate.add_argument("--generation-examples", type=int)
    evaluate.add_argument("--no-track", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    export = subparsers.add_parser("export", parents=[common], help="Export or merge the adapter")
    export.add_argument("--adapter")
    export.add_argument("--output")
    export.add_argument("--merged", action="store_true", help="Merge LoRA weights into the base")
    export.set_defaults(func=command_export)

    serve = subparsers.add_parser(
        "serve", parents=[common], help="Run the FastAPI inference server"
    )
    serve.add_argument("--adapter")
    serve.add_argument("--base-model")
    serve.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=command_serve)

    ui = subparsers.add_parser("mlflow-ui", parents=[common], help="Open the local MLflow UI")
    ui.add_argument("--port", type=int, default=5000)
    ui.set_defaults(func=command_mlflow_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    if args.command == "serve":
        config = load_config(args.config)
        args.host = args.host or config.serving.host
        args.port = args.port or config.serving.port
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    except ValueError as exc:
        LOGGER.error("Invalid configuration or dataset: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
