"""MLflow experiment tracking.

MLflow is optional: when the ``train`` extra is missing or tracking is
disabled, :func:`build_tracker` returns a no-op tracker so training and
evaluation still run. Failures to talk to the tracking store are raised rather
than swallowed, because a silently missing run is worse than a loud one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from loraflow.config import AppConfig

LOGGER = logging.getLogger("loraflow.tracking")

_MAX_PARAM_CHARS = 250


def flatten(prefix: str, payload: Any, out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn a nested config/report dict into MLflow-friendly dotted keys."""

    out = {} if out is None else out
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flatten(name, value, out)
    elif isinstance(payload, (list, tuple)):
        # Lists are summarised rather than exploded: a long target_modules list
        # or a list of example strings would flood the parameter table.
        out[prefix] = f"{len(payload)} items: " + ", ".join(str(item)[:40] for item in payload[:6])
    elif payload is None or isinstance(payload, (bool, int, float, str)):
        out[prefix] = str(payload)[:_MAX_PARAM_CHARS] if isinstance(payload, str) else payload
    else:
        out[prefix] = str(payload)[:_MAX_PARAM_CHARS]
    return out


_METRIC_TYPES = (int, float)

# Numeric scalars worth charting. Anything else numeric (a seed, a port) stays
# a parameter so the metric list keeps meaning.
_METRIC_MARKERS = (
    "loss",
    "accuracy",
    "perplexity",
    "runtime_seconds",
    "steps",
    "epochs_completed",
    "exact_match",
    "prefix_match",
    "parameters",
    "trainable_fraction",
    "tokens",
    "examples",
    "chars",
    "predictions",
)


def _metrics_only(flat: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, bool) or not isinstance(value, _METRIC_TYPES):
            continue
        if any(marker in key for marker in _METRIC_MARKERS):
            metrics[key.replace("/", "_").replace(" ", "_")] = float(value)
    return metrics


class NoopTracker:
    """Used when tracking is disabled or MLflow is unavailable."""

    enabled = False

    def start(self, *, run_name: str | None = None) -> None:
        LOGGER.info("MLflow tracking disabled")

    def log_run(self, config: AppConfig, *, extra: dict[str, Any] | None = None) -> None:
        return None

    def log_training(self, metadata: dict[str, Any]) -> None:
        return None

    def log_evaluation(self, report: dict[str, Any]) -> None:
        return None

    def log_adapter(self, adapter_dir: Path) -> None:
        return None

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        return None

    def finish(self) -> None:
        return None

    @property
    def run_id(self) -> str | None:
        return None


class MLflowTracker:
    """Logs LoRAFlow configs, metrics and artifacts into one MLflow run."""

    enabled = True

    def __init__(self, config: AppConfig, *, tags: dict[str, str] | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        self.config = config
        self.tags = dict(tags or {})
        self._run: Any = None
        self._owns_run = False

        tracking_uri = os.environ.get(
            "LORAFLOW_MLFLOW_TRACKING_URI", config.tracking.tracking_uri
        )
        # A relative sqlite/file URI resolves against the process working
        # directory; normalise it so `mlflow ui` finds the store from anywhere.
        if tracking_uri.startswith("file://"):
            path = Path(tracking_uri.removeprefix("file://")).expanduser()
            tracking_uri = "file://" + str(path.resolve())
        elif tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
            relative = tracking_uri.removeprefix("sqlite:///")
            if relative:
                tracking_uri = "sqlite:///" + str(Path(relative).resolve())
        mlflow.set_tracking_uri(tracking_uri)

        self.artifact_dir: Path | None = None
        client = mlflow.MlflowClient()
        experiment_name = config.tracking.experiment_name
        desired_location: str | None = None
        artifacts_uri = os.environ.get(
            "LORAFLOW_MLFLOW_ARTIFACTS_DIR", config.tracking.artifacts_dir
        )
        if artifacts_uri:
            self.artifact_dir = (
                Path(artifacts_uri.removeprefix("file://")).expanduser().resolve()
            )
            desired_location = "file://" + str(self.artifact_dir)

        # MLflow 3.x fixes the artifact location when an experiment is created,
        # and create_experiment raises on a name collision, so look it up first.
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            client.create_experiment(experiment_name, artifact_location=desired_location)
        elif desired_location and existing.artifact_location.rstrip("/") != desired_location.rstrip(
            "/"
        ):
            LOGGER.warning(
                "Experiment %r already exists with artifact location %s; the "
                "configured artifacts_dir (%s) is ignored. MLflow only accepts an "
                "artifact location when an experiment is created.",
                experiment_name,
                existing.artifact_location,
                desired_location,
            )
        mlflow.set_experiment(experiment_name)

    def start(self, *, run_name: str | None = None) -> None:
        if self._run is not None:
            return
        self._run = self._mlflow.start_run(
            run_name=run_name or self.config.tracking.run_name,
        )
        self._owns_run = True
        if self.tags:
            self._mlflow.set_tags(self.tags)
        LOGGER.info(
            "MLflow run %s in experiment %r",
            self._run.info.run_id,
            self.config.tracking.experiment_name,
        )

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run is not None else None

    def log_run(self, config: AppConfig, *, extra: dict[str, Any] | None = None) -> None:
        self.start()
        payload = {
            "data": config.data.model_dump(mode="json"),
            "lora": config.lora.model_dump(mode="json"),
            "training": config.training.model_dump(mode="json"),
            "evaluation": config.evaluation.model_dump(mode="json"),
            "export": config.export.model_dump(mode="json"),
            "serving": config.serving.model_dump(mode="json"),
        }
        if extra:
            payload.update(extra)
        self._log_flat(payload)

    def log_training(self, metadata: dict[str, Any]) -> None:
        self.start()
        self._log_flat({k: v for k, v in metadata.items() if k != "examples"})
        self.log_artifact(Path(self.config.training.output_dir) / "training_metadata.json")

    def log_evaluation(self, report: dict[str, Any]) -> None:
        self.start()
        slim = {k: v for k, v in report.items() if k not in ("examples",)}
        self._log_flat(slim, prefix="eval")
        self.log_artifact(self.config.evaluation.output_file)

    def log_adapter(self, adapter_dir: str | Path) -> None:
        """Upload the adapter weights themselves, not just their location."""

        if not self.config.tracking.log_artifacts:
            return
        self.start()
        adapter_dir = Path(adapter_dir)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = adapter_dir / name
            if path.is_file():
                self._mlflow.log_artifact(str(path), artifact_path="adapter")
            else:
                LOGGER.warning("Expected adapter file missing: %s", path)
        metadata = Path(self.config.training.output_dir) / "training_metadata.json"
        if metadata.is_file():
            self._mlflow.log_artifact(str(metadata), artifact_path="adapter")

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        if not self.config.tracking.log_artifacts:
            return
        self.start()
        path = Path(path)
        if path.is_file():
            self._mlflow.log_artifact(str(path), artifact_path=name)

    def _log_flat(self, payload: dict[str, Any], *, prefix: str = "") -> None:
        flat = flatten(prefix, payload)
        metrics = _metrics_only(flat)
        params = {key: value for key, value in flat.items() if key not in metrics}
        # MLflow 3.x has no fluent log_batch, so use the two dedicated calls.
        if params:
            self._mlflow.log_params(
                {key: str(value)[:_MAX_PARAM_CHARS] for key, value in params.items()}
            )
        if metrics:
            self._mlflow.log_metrics(metrics)

    def finish(self) -> None:
        if self._run is not None and self._owns_run:
            self._mlflow.end_run()
            self._run = None


def build_tracker(config: AppConfig, *, tags: dict[str, str] | None = None) -> Any:
    """Return an MLflow tracker, or a no-op when disabled or unavailable."""

    if not config.tracking.enabled:
        return NoopTracker()
    try:
        return MLflowTracker(config, tags=tags)
    except ImportError as exc:
        LOGGER.warning(
            "MLflow tracking requested but unavailable (%s); install the 'train' "
            "extra. Continuing without tracking.",
            exc,
        )
        return NoopTracker()
