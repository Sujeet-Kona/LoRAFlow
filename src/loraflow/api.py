"""FastAPI inference service for a base model plus a trained LoRA adapter.

The app accepts an already-built service rather than loading weights itself, so
the HTTP layer is testable without downloading a model.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from peft import PeftModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM

from loraflow.config import AppConfig
from loraflow.data import render_prompt
from loraflow.runtime import cpu_threads, resolve_device, resolve_dtype, set_global_seed
from loraflow.training import ADAPTER_DIRNAME, _dtype_kwarg_name, load_tokenizer

LOGGER = logging.getLogger("loraflow.api")



def _validate_prompt_token_length(token_count: int, max_length: int) -> None:
    """Reject oversized prompts instead of silently dropping their leading context."""
    if token_count > max_length:
        raise ValueError(
            f"Prompt has {token_count} tokens, above the configured max_seq_length "
            f"of {max_length}; shorten the prompt or increase the limit."
        )

REQUESTS = Counter("loraflow_requests_total", "Inference requests", ["endpoint", "status"])
LATENCY = Histogram(
    "loraflow_generation_seconds",
    "Generation latency",
    ["endpoint"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 40, 80),
)
IN_FLIGHT = Gauge("loraflow_requests_in_flight", "Requests currently being served")
MODEL_LOADED = Gauge("loraflow_model_loaded", "1 when the model is resident")


class GenerateRequest(BaseModel):
    """Request body for ``/generate``.

    ``prompt`` maps onto the dataset's instruction field and ``input`` onto its
    input field, so a caller can reproduce the exact prompt the adapter was
    trained on. Bounds are enforced here so malformed input becomes an HTTP 422
    before any tensor is allocated.
    """

    prompt: str = Field(min_length=1, max_length=8000)
    input: str | None = Field(default=None, max_length=8000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    format_prompt: bool = True


class ServiceLike(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...


class ModelService:
    """Owns one loaded model and bounds concurrent generations."""

    def __init__(
        self,
        config: AppConfig,
        *,
        adapter_dir: str | Path | None = None,
        base_model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.base_model = base_model or config.training.base_model
        self.adapter_dir = (
            Path(adapter_dir) if adapter_dir else Path(config.training.output_dir) / ADAPTER_DIRNAME
        )
        self.device = device or resolve_device(config.serving.device)
        self.dtype = resolve_dtype(self.device, config.training.dtype)
        self._slots = threading.Semaphore(config.serving.max_concurrent_requests)
        self.model: Any = None
        self.tokenizer: Any = None
        self.adapter_applied = False
        self.load_error: str | None = None

    def load(self) -> None:
        """Load base weights and attach the adapter when one exists."""

        if self.model is not None:
            return
        torch.set_num_threads(cpu_threads())
        set_global_seed(self.config.training.seed, self.device)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, **{_dtype_kwarg_name(): self.dtype}
            )
            tokenizer = load_tokenizer(self.base_model)
            if self.adapter_dir.is_dir():
                model = PeftModel.from_pretrained(model, str(self.adapter_dir))
                tokenizer = load_tokenizer(self.adapter_dir)
                self.adapter_applied = True
            else:
                LOGGER.warning("No adapter at %s; serving the base model", self.adapter_dir)
            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            self.load_error = None
            MODEL_LOADED.set(1)
            LOGGER.info(
                "loaded base_model=%s adapter_applied=%s device=%s",
                self.base_model,
                self.adapter_applied,
                self.device,
            )
        except Exception as exc:
            MODEL_LOADED.set(0)
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("model loading failed")
            raise

    def generate(
        self,
        prompt: str,
        *,
        input_text: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        format_prompt: bool = True,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        text = self._prepare_prompt(prompt, input_text=input_text, format_prompt=format_prompt)
        if len(text) > self.config.serving.max_input_chars:
            raise ValueError(
                f"Prompt is {len(text)} characters, above the configured limit "
                f"of {self.config.serving.max_input_chars}"
            )
        new_tokens_limit = max_new_tokens or self.config.serving.max_new_tokens
        do_sample = temperature > 0

        started = time.perf_counter()
        with self._slots:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_length = int(encoded["input_ids"].size(1))
            _validate_prompt_token_length(prompt_length, self.config.data.max_seq_length)
            encoded = encoded.to(self.device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": new_tokens_limit,
                "do_sample": do_sample,
                "pad_token_id": (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            with torch.no_grad():
                output_ids = self.model.generate(**encoded, **kwargs)
            completion = self.tokenizer.decode(
                output_ids[0][prompt_length:], skip_special_tokens=True
            )
        return {
            "completion": completion.strip(),
            # The task's answer is one line, but generations occasionally run on
            # without emitting EOS. Both are returned so the raw output stays
            # visible rather than being quietly truncated.
            "completion_line": completion.strip().splitlines()[0].strip()
            if completion.strip()
            else "",
            "prompt_tokens": int(prompt_length),
            "completion_tokens": int(output_ids.size(1) - prompt_length),
            "adapter_applied": self.adapter_applied,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }

    def _prepare_prompt(
        self, prompt: str, *, input_text: str | None = None, format_prompt: bool = True
    ) -> str:
        """Rebuild the training prompt from the caller's instruction and input.

        A prompt that already carries the template header is passed through
        untouched, so callers can also send a fully formatted prompt.
        """

        if not format_prompt or prompt.lstrip().startswith("### Instruction:"):
            return prompt
        return render_prompt(
            prompt.strip(),
            input_text=(input_text or "").strip(),
            template=self.config.data.prompt_template,
        )

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "adapter_applied": self.adapter_applied,
            "device": self.device,
            "dtype": str(self.dtype),
            "load_error": self.load_error,
            "max_new_tokens": self.config.serving.max_new_tokens,
        }


def create_app(
    config: AppConfig,
    service: ServiceLike | None = None,
    *,
    load_on_startup: bool = False,
) -> FastAPI:
    """Build the inference app around ``service``."""

    app = FastAPI(
        title="LoRAFlow inference",
        version="0.1.0",
        description="Serve a base model with a LoRA adapter attached.",
    )
    app.state.config = config
    app.state.service = service

    if load_on_startup and service is not None:
        try:
            service.load()  # type: ignore[attr-defined]
        except Exception:
            # Surfaced through /health and /generate instead of aborting boot,
            # so the process stays observable when weights are unavailable.
            LOGGER.error("Startup load failed; /health will report the error")

    @app.middleware("http")
    async def _instrument(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        IN_FLIGHT.inc()
        try:
            response = await call_next(request)
        finally:
            IN_FLIGHT.dec()
        path = request.url.path
        if path in {"/health", "/generate", "/metrics"}:
            REQUESTS.labels(path, str(response.status_code)).inc()
            if path == "/generate":
                LATENCY.labels(path).observe(time.perf_counter() - started)
        response.headers["X-Process-Milliseconds"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.get("/health")
    def health() -> JSONResponse:
        state = service.status() if service is not None else {"loaded": False}
        if state.get("load_error"):
            return JSONResponse(
                {"status": "degraded", "service": state},
                status_code=503,
            )
        if not state.get("loaded"):
            return JSONResponse(
                {"status": "no_model", "service": state},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "service": state}, status_code=200)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate")
    def generate(payload: GenerateRequest) -> JSONResponse:
        if service is None:
            return JSONResponse({"detail": "No model service is attached"}, status_code=503)
        if not service.status().get("loaded"):
            return JSONResponse(
                {"detail": "Model unavailable: not loaded"},
                status_code=503,
            )
        try:
            result = service.generate(
                payload.prompt,
                input_text=payload.input,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                format_prompt=payload.format_prompt,
            )
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            LOGGER.exception("generation failed")
            return JSONResponse(
                {"detail": f"Inference failed: {type(exc).__name__}: {exc}"},
                status_code=500,
            )
        LOGGER.info(
            "generated %d tokens in %.3fs adapter=%s",
            result.get("completion_tokens", 0),
            result.get("elapsed_seconds", 0.0),
            result.get("adapter_applied"),
        )
        return JSONResponse(result, status_code=200)

    return app


def build_service(config: AppConfig, *, adapter_dir: str | Path | None = None) -> ModelService:
    """Construct and load a service, for CLI and server entrypoints."""

    service = ModelService(config, adapter_dir=adapter_dir)
    service.load()
    return service
