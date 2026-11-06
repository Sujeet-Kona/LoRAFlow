# LoRAFlow dev image: CPU PyTorch, since this project targets a local
# laptop path and explicitly excludes CUDA-only infrastructure.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface

WORKDIR /app

# CPU wheels keep the image small; swap the index URL for the CUDA build only
# on a GPU host.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.15.0+cpu"

COPY pyproject.toml README.md ./
COPY src ./src

# Install the package plus the train and serve extras against the torch already
# present, so transformers/peft/trl reuse it rather than pulling a CUDA build.
RUN pip install -e ".[train,serve]"

COPY configs ./configs
COPY scripts ./scripts
# Dataset files are generated or mounted at runtime; they are intentionally not in Git.

EXPOSE 8000

# Default to the API; run training with:
#   docker compose run --rm train
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["loraflow", "serve", "--config", "configs/train.yaml", "--host", "0.0.0.0", "--port", "8000"]
