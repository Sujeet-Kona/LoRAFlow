"""Device, dtype and seeding helpers shared by training, evaluation and serving.

Importing this module requires torch, unlike :mod:`loraflow.data`.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DEVICE_NAMES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "auto") -> str:
    """Pick a concrete torch device name.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. An explicitly requested
    accelerator raises instead of silently falling back, so a config that says
    ``cuda`` on a Mac fails loudly rather than training somewhere unexpected.
    """

    if requested not in DEVICE_NAMES:
        raise ValueError(f"Unknown device {requested!r}; expected one of {DEVICE_NAMES}")

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested but CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' requested but MPS is not available")
    return requested


def resolve_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """Map a config dtype onto a torch dtype that the device actually supports.

    ``auto`` selects bf16 on CUDA only. On MPS and CPU it stays fp32: fp16
    training is unstable on those paths and bf16 support on MPS is partial, so
    fp32 is the safe default for a 16 GB Apple Silicon laptop.
    """

    if requested not in ("auto", "fp32", "bf16", "fp16"):
        raise ValueError(f"Unknown dtype {requested!r}")

    if requested == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32

    mapping = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = mapping[requested]
    if requested in ("bf16", "fp16") and device == "cpu":
        raise RuntimeError(
            f"{requested.upper()} training is not supported on CPU; use dtype 'fp32'"
        )
    return dtype


def set_global_seed(seed: int, device: str = "cpu") -> None:
    """Seed Python, NumPy when installed, and the active torch device."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) is deliberately not enabled: it
    # is CUDA-centric and raises on unsupported MPS ops. The seed above plus
    # greedy decoding is what makes a single-machine rerun reproducible.


def cpu_threads() -> int:
    """Reasonable thread count for a laptop: half the cores, at least 1."""

    count = os.cpu_count() or 4
    return max(1, count // 2)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically-ish (temp file then rename) and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=json_default)
        file.write("\n")
    tmp.replace(target)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
