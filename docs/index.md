# AI Knowledge Lakehouse

A multi-source RAG platform: PDFs, Markdown, HTML, and GitHub repositories flow through a
Bronze/Silver/Gold lakehouse into hybrid (dense + sparse) retrieval and a citation-checked chat
API, orchestrated by Airflow and observable end to end.

```
Sources (PDF/MD/HTML/GitHub)
    -> Bronze (raw, content-addressed, immutable)
    -> Silver (parsed, validated, deduplicated, PII-scanned)
    -> Gold (chunked, embedded, retrieval-ready)
    -> Qdrant (dense) + BM25 (sparse) -> RRF fusion -> reranking
    -> FastAPI (search / chat / upload / admin) -> answers with citations
```

## Where to start

- **Running it locally**: see the root `README.md` and `make help`.
- **How a request flows through the system**: [Architecture Decisions](adr/ADR-013-airflow-isolated-akl-venv.md)
  cover the two decisions with the widest blast radius (why Airflow tasks run in an isolated
  virtualenv, and how error codes are chosen).
- **Something's on fire**: [Incident response](runbooks/incident-response.md).
- **What's instrumented**: [Metrics reference](reference/metrics.md) and the four Grafana
  dashboards under the `AKL` folder (`make grafana-up`).
- **What can go wrong and what code it returns**: [Error code reference](reference/errors.md).
- **Is retrieval any good, and is it fast enough**: [Benchmarks](benchmarks/index.md) and
  `make eval-run` / `make bench`.
- **Cutting a release**: [Release checklist](release-checklist.md).

## Layout

| Path | What lives there |
|---|---|
| `akl/` | The application: ingestion, chunking, embedding, retrieval, RAG, API, governance, observability, eval |
| `airflow/` | DAGs and the Airflow-side plugin code (thin — see ADR-013) |
| `configs/` | Runtime YAML settings and DAG configuration |
| `docker/` | Dockerfiles and service configs (Postgres init, Traefik, Prometheus, Grafana) |
| `tests/unit` `tests/component` | Fast, no-dependency tests / real-service tests (Postgres, MinIO, Qdrant) |
| `tests/load` `tests/chaos` | Locust load test / opt-in destructive chaos tests against a live stack |
| `docs/adr` | Architecture decision records for the choices worth explaining later |
