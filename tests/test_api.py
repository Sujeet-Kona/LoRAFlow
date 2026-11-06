"""HTTP-layer tests using a stub service, so no weights are downloaded."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from loraflow.api import create_app  # noqa: E402
from loraflow.config import AppConfig, DataConfig, TrainingConfig  # noqa: E402


class FakeService:
    def __init__(
        self, *, loaded: bool = True, error: str | None = None, raises: Exception | None = None
    ):
        self._loaded = loaded
        self._error = error
        self._raises = raises
        self.calls: list[dict] = []

    def load(self) -> None:
        if self._error:
            raise RuntimeError(self._error)

    def generate(self, prompt: str, **kwargs) -> dict:
        self.calls.append({"prompt": prompt, **kwargs})
        if self._raises is not None:
            raise self._raises
        return {
            "completion": f"sev=routine cat=skin act=log_only :: {len(prompt)}",
            "prompt_tokens": 7,
            "completion_tokens": 9,
            "adapter_applied": True,
            "elapsed_seconds": 0.01,
        }

    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "base_model": "fake/model",
            "adapter_applied": True,
            "device": "cpu",
            "dtype": "torch.float32",
            "load_error": self._error,
            "max_new_tokens": 32,
        }


def make_config(**overrides) -> AppConfig:
    config = AppConfig(
        data=DataConfig(train_file="data/train.jsonl"),
        training=TrainingConfig(base_model="fake/model"),
    )
    config.serving.max_input_chars = 600
    config.serving.max_new_tokens = 32
    for key, value in overrides.items():
        setattr(config.serving, key, value)
    return config


def client_for(service) -> TestClient:
    return TestClient(create_app(make_config(), service))


def test_health_reports_ok_when_loaded() -> None:
    response = client_for(FakeService()).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"]["adapter_applied"] is True


def test_health_is_503_without_service() -> None:
    response = TestClient(create_app(make_config(), None)).get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "no_model"


def test_health_reports_degraded_after_load_error() -> None:
    service = FakeService(loaded=False, error="OSError: weights missing")

    response = client_for(service).get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_generate_returns_completion() -> None:
    service = FakeService()
    response = client_for(service).post("/generate", json={"prompt": "Label this note"})

    assert response.status_code == 200
    body = response.json()
    assert body["completion"].startswith("sev=routine")
    assert body["completion_tokens"] == 9
    assert service.calls[0]["prompt"] == "Label this note"


def test_generate_passes_the_prompt_through_to_the_service() -> None:
    # Formatting is the service's job, not the endpoint's.
    service = FakeService()
    client_for(service).post("/generate", json={"prompt": "Label this note"})

    assert service.calls[0]["prompt"] == "Label this note"


def test_model_service_wraps_plain_text_in_the_training_template() -> None:
    from loraflow.api import ModelService

    service = ModelService(make_config())
    formatted = service._prepare_prompt("Label this note", format_prompt=True)

    assert formatted.startswith("### Instruction:")
    assert formatted.endswith("### Response:\n")
    assert "Label this note" in formatted


def test_model_service_leaves_an_already_formatted_prompt_alone() -> None:
    from loraflow.api import ModelService

    service = ModelService(make_config())
    already = "### Instruction:\nLabel\n### Response:\n"

    assert service._prepare_prompt(already, format_prompt=True) == already
    assert service._prepare_prompt("raw text", format_prompt=False) == "raw text"


def test_model_service_puts_input_in_its_own_block() -> None:
    from loraflow.api import ModelService

    service = ModelService(make_config())
    formatted = service._prepare_prompt("Label the note", input_text="Animal A1: lame")

    assert "### Instruction:\nLabel the note\n" in formatted
    assert "### Input:\nAnimal A1: lame\n\n### Response:\n" in formatted


def test_generate_accepts_a_separate_input_field() -> None:
    service = FakeService()
    client_for(service).post(
        "/generate", json={"prompt": "Label the note", "input": "Animal A1: lame"}
    )

    assert service.calls[0]["input_text"] == "Animal A1: lame"


def test_generate_passes_through_preformatted_prompt() -> None:
    service = FakeService()
    already = "### Instruction:\nLabel\n### Response:\n"

    client_for(service).post(
        "/generate",
        json={"prompt": already, "format_prompt": False},
    )

    assert service.calls[0]["prompt"] == already


def test_generate_forwards_sampling_parameters() -> None:
    service = FakeService()
    client_for(service).post(
        "/generate",
        json={"prompt": "hi", "max_new_tokens": 12, "temperature": 0.7, "top_p": 0.9},
    )

    call = service.calls[0]
    assert call["max_new_tokens"] == 12
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.9


def test_generate_rejects_empty_prompt_with_422() -> None:
    response = client_for(FakeService()).post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_rejects_oversized_max_new_tokens() -> None:
    response = client_for(FakeService()).post(
        "/generate", json={"prompt": "hi", "max_new_tokens": 5000}
    )
    assert response.status_code == 422


def test_generate_maps_value_error_to_400() -> None:
    service = FakeService(raises=ValueError("Prompt is 900 characters, above the limit"))

    response = client_for(service).post("/generate", json={"prompt": "x" * 50})

    assert response.status_code == 400
    assert "above the limit" in response.json()["detail"]


def test_generate_maps_runtime_error_to_503() -> None:
    service = FakeService(raises=RuntimeError("Model is not loaded"))

    response = client_for(service).post("/generate", json={"prompt": "hi"})

    assert response.status_code == 503


def test_generate_maps_unexpected_failure_to_500() -> None:
    service = FakeService(raises=IndexError("kernel blew up"))

    response = client_for(service).post("/generate", json={"prompt": "hi"})

    assert response.status_code == 500
    assert "IndexError" in response.json()["detail"]


def test_generate_refuses_when_not_loaded() -> None:
    service = FakeService(loaded=False)

    response = client_for(service).post("/generate", json={"prompt": "hi"})

    assert response.status_code == 503
    assert service.calls == []


def test_metrics_endpoint_exposes_prometheus_counters() -> None:
    client = client_for(FakeService())
    client.get("/health")
    client.post("/generate", json={"prompt": "hi"})

    body = client.get("/metrics").text

    assert "loraflow_requests_total" in body
    assert "loraflow_generation_seconds" in body
    assert "loraflow_model_loaded" in body


def test_response_carries_processing_time_header() -> None:
    response = client_for(FakeService()).get("/health")
    assert "X-Process-Milliseconds" in response.headers


def test_unknown_route_is_404() -> None:
    assert client_for(FakeService()).get("/does-not-exist").status_code == 404



def test_prompt_token_limit_rejects_truncation() -> None:
    from loraflow.api import _validate_prompt_token_length

    with pytest.raises(ValueError, match="above the configured max_seq_length"):
        _validate_prompt_token_length(257, 256)


def test_prompt_token_limit_accepts_exact_boundary() -> None:
    from loraflow.api import _validate_prompt_token_length

    _validate_prompt_token_length(256, 256)
