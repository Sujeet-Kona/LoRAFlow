"""Tests for the device/dtype/JSON runtime layer."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from loraflow.runtime import (  # noqa: E402
    cpu_threads,
    json_default,
    read_json,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    write_json,
)


def test_resolve_device_auto_returns_a_real_device() -> None:
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_resolve_device_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_resolve_device_raises_for_unavailable_accelerator() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present, so the failure path cannot be exercised")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_resolve_device_mps_only_when_supported() -> None:
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS is not available"):
            resolve_device("mps")
    else:
        assert resolve_device("mps") == "mps"


def test_resolve_dtype_auto_keeps_cpu_on_fp32() -> None:
    assert resolve_dtype("cpu", "auto") == torch.float32
    assert resolve_dtype("cuda", "auto") == torch.bfloat16


def test_resolve_dtype_rejects_low_precision_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="not supported on CPU"):
        resolve_dtype("cpu", "bf16")


def test_resolve_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("cpu", "int8")


def test_set_global_seed_makes_torch_draws_repeatable() -> None:
    set_global_seed(123, "cpu")
    first = torch.rand(4)
    set_global_seed(123, "cpu")
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_cpu_threads_is_positive_and_bounded() -> None:
    threads = cpu_threads()
    assert 1 <= threads <= (os_cpu_count() or threads)


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 4


def test_write_and_read_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    payload = {"b": 2, "a": 1, "path": Path("x/y"), "dtype": torch.float32, "items": [1, 2]}

    written = write_json(path, payload)
    loaded = read_json(written)

    assert loaded == {"a": 1, "b": 2, "path": "x/y", "dtype": "torch.float32", "items": [1, 2]}
    # Keys are sorted so repeated writes are diff-friendly.
    assert list(loaded) == ["a", "b", "dtype", "items", "path"]


def test_write_json_replaces_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"first": True})
    write_json(path, {"second": True})

    assert read_json(path) == {"second": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_json_default_rejects_unserialisable_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serialisable"):
        json_default(object())


def test_write_json_failure_is_visible(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError):
        write_json(blocked / "report.json", {"a": 1})


def test_set_global_seed_makes_numpy_draws_repeatable() -> None:
    set_global_seed(321, "cpu")
    first = np.random.random(4)
    set_global_seed(321, "cpu")
    second = np.random.random(4)
    assert np.array_equal(first, second)
