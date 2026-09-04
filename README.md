# Enterprise AI Knowledge Lakehouse (AKL)

Production-scale, multi-source AI knowledge platform built on a Bronze/Silver/Gold
lakehouse with hybrid retrieval and cited RAG answers.

**Stack:** Python 3.12 · FastAPI · DuckDB · Parquet · MinIO · Qdrant · Apache Airflow ·
PostgreSQL · BAAI/bge-small-en-v1.5 · MLflow · Prometheus · Grafana · OpenTelemetry · Docker Compose

> The complete engineering design is in [`PRD.md`](./PRD.md). It is the source of truth
> for architecture, schemas, APIs, DAGs, metrics and error codes. Code follows the PRD;
> deviations are recorded as ADRs under `docs/adr/`.

## What it does

1. Ingests PDF, Markdown, HTML and GitHub repositories into an immutable **Bronze** layer.
2. Parses, cleans, deduplicates and chunks documents into **Silver** Parquet datasets.
3. Produces AI-ready **Gold** retrieval units and cached embeddings.
4. Reconciles a **Qdrant** vector index and a **BM25** index as derived state.
5. Answers questions via **hybrid retrieval → cross-encoder rerank → cited generation**.
6. Orchestrates everything with five **Airflow** DAGs, fully observable in Grafana.

## Repository layout

| Path | Contents |
|---|---|
| `akl/` | Python package (lakehouse, ingestion, chunking, embedding, rag, api, …) |
| `airflow/` | DAGs, plugins and Airflow config |
| `configs/` | YAML configuration and prompt templates |
| `docker/`, `docker-compose*.yml` | Container images and Compose stack |
| `observability/` | Prometheus rules, Grafana dashboards, OTel collector config |
| `tests/` | Unit, component, integration, API, eval and load tests |
| `docs/` | ADRs, runbooks, benchmarks |

## Quickstart (developer)

```bash
# Prerequisites: Python 3.12, uv, Docker Desktop (or Docker Engine 24+)
uv sync --extra dev          # install project + dev tools into .venv
make hooks                   # install pre-commit hooks
make lint && make test       # verify the toolchain
make up                      # start the stack (Milestone 2)
```

See `make help` for every target.

## Status

Under active construction, milestone by milestone. See `PRD.md` Chapter 16 for the roadmap.

## License

MIT
