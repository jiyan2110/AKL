# Changelog

All notable changes to the Enterprise AI Knowledge Lakehouse are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] — Initial release

The full 58-milestone build: ingestion (PDF/Markdown/HTML/GitHub) → chunking → embeddings →
hybrid (dense + BM25) retrieval with reranking → a FastAPI gateway (search, streaming chat,
uploads, admin) → Airflow orchestration → observability (structured logging, metrics, tracing,
lineage, Grafana dashboards, alerting) → governance (RBAC, audit log, PII scanning, GDPR
erasure, hard delete) → an eval/load/chaos/benchmark harness → CI/CD and a production Docker
Compose + Traefik/TLS deployment path.

### Added
- Bronze/Silver/Gold lakehouse on DuckDB + Parquet + MinIO, with schema evolution, compaction,
  and dataset lineage.
- Connectors for Markdown, PDF, HTML, and GitHub, with quality gates, PII scanning, dedup, and
  quarantine.
- Hierarchical/semantic/token chunking with stable identity, incremental re-chunking, and lineage.
- An embedding pipeline (BGE via ONNX, with a deterministic offline provider for tests) with a
  content-addressed cache, and Qdrant collection management with drift-verified reconciliation.
- Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused via RRF, cross-encoder reranking (with a
  deterministic lexical fallback), a context builder, and citation-checked answers.
- A FastAPI gateway: auth (JWT + API keys), rate limiting, search/chat (streaming) endpoints,
  document upload/list/detail/delete (soft and hard), conversation memory, health/readiness.
- Airflow DAGs for ingestion, chunking, embedding, Qdrant sync, and maintenance, running the same
  task entrypoints the CLI uses (`akl-cli pipeline ...`), so every stage works with or without
  Airflow.
- Observability: structured JSON logging with secret redaction, a full Prometheus metric catalog
  (scraped + Pushgateway-pushed), OpenTelemetry tracing, dataset/document lineage, four Grafana
  dashboards, Prometheus alert rules, optional MLflow logging.
- Governance: role-scoped permissions, an append-only audit log, PII detection (hashed-only
  storage), GDPR erasure/export, a genuinely destructive hard-delete path, and a production
  Docker Compose + Traefik/TLS deployment.
- An evaluation harness (synthetic QA generation, recall/MRR/nDCG/refusal metrics, confidence
  calibration), a Locust load test, opt-in chaos tests, and an in-process latency benchmark.
- CI (lint/unit/component/DAG-integrity/Docker build), a nightly integration+eval run, a
  multi-arch signed-image release workflow, and a deploy template.

[Unreleased]: https://github.com/your-org/ai-knowledge-lakehouse/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/ai-knowledge-lakehouse/releases/tag/v1.0.0
