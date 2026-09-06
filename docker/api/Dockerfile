# AKL API — production image (PRD §11.9). Multi-stage: build the venv, then a slim runtime layer.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY akl ./akl
COPY configs ./configs
RUN python -m venv /opt/akl-venv \
 && /opt/akl-venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/akl-venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim
RUN groupadd --gid 10001 akl && useradd --uid 10001 --gid akl --create-home --shell /usr/sbin/nologin akl \
 && apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/akl-venv /opt/akl-venv
COPY --chown=akl:akl configs /app/configs
WORKDIR /app
ENV PATH="/opt/akl-venv/bin:$PATH" \
    AKL_ENV=prod \
    AKL_CONFIG_DIR=/app/configs \
    AKL_MODELS_DIR=/models \
    PYTHONUNBUFFERED=1
USER akl
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1
# Single worker: one DuckDB engine per process (see akl/api/deps.py AppState.lock); scale by replicas, not workers.
CMD ["uvicorn", "akl.api.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
