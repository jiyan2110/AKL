# Enterprise AI Knowledge Lakehouse

**Production-scale Multi-Source AI Knowledge Platform using Lakehouse Architecture, Vector Search, RAG, Airflow, DuckDB, Parquet, MinIO, Qdrant, FastAPI, MLflow, Prometheus and Grafana.**

| Field | Value |
|---|---|
| Document type | Product Requirements Document / Engineering Design Specification |
| Document ID | AKL-PRD-001 |
| Status | Approved for implementation |
| Version | 1.0.0 |
| Python package name | `akl` |
| Repository name | `ai-knowledge-lakehouse` |
| Primary implementer | Claude Code / GitHub Copilot (agentic implementation from this specification) |
| Target reader | Data Engineers, AI Engineers, Platform Engineers |

---

## Table of Contents

- Chapter 0 — Executive Summary
- Chapter 1 — System Architecture
- Chapter 2 — Lakehouse Design
- Chapter 3 — Document Ingestion Engine
- Chapter 4 — Chunking Engine
- Chapter 5 — Embedding Pipeline
- Chapter 6 — RAG Engine
- Chapter 7 — Airflow Orchestration
- Chapter 8 — Observability & DataOps
- Chapter 9 — Security & Governance
- Chapter 10 — Backend APIs
- Chapter 11 — Infrastructure
- Chapter 12 — Testing Strategy
- Chapter 13 — CI/CD
- Chapter 14 — PB Scale Evolution
- Chapter 15 — Repository Blueprint
- Chapter 16 — Implementation Roadmap
- Appendix A — SQL Schemas
- Appendix B — Environment Variables
- Appendix C — Airflow DAG Config
- Appendix D — Qdrant Collection Config
- Appendix E — Docker Compose Specification
- Appendix F — Metrics Catalog
- Appendix G — Error Catalog
- Appendix H — Glossary

---

## Conventions Used in This Document

| Convention | Meaning |
|---|---|
| **MUST / MUST NOT** | Hard requirement. Implementation is non-compliant without it. |
| **SHOULD** | Strong recommendation; deviation requires a documented reason in an ADR. |
| **MAY** | Optional. |
| `Current MVP` | Behaviour required for the local Docker Compose deployment. |
| `Enterprise Scale` | Design intent for how the same component behaves at 100 TB – 1 PB. Not implemented in MVP, but the MVP design MUST NOT preclude it. |
| `akl.*` | Python package path inside the repository. |
| `AKL_*` | Environment variable namespace. |
| `akl_*` | Prometheus metric namespace. |
| `AKL-Exxxx` | Error code namespace (Appendix G). |
| `ADR-xxx` | Architecture Decision Record (Chapter 1.9). |

All identifiers, table names, bucket names, DAG IDs, metric names, and error codes defined in this document are **normative**. Implementers MUST use them exactly as written so that the appendices, tests, dashboards and runbooks stay consistent.

---

# Chapter 0 — Executive Summary

## 0.1 Vision

Every engineering organisation accumulates knowledge across heterogeneous surfaces: GitHub repositories, PDF specifications, Markdown runbooks, HTML wikis and vendor documentation. That knowledge is fragmented, versioned inconsistently, and unsearchable across boundaries. Engineers repeatedly rediscover facts that already exist in documents they cannot find.

The Enterprise AI Knowledge Lakehouse (hereafter **AKL**) is an internal platform that treats organisational documents as a **data engineering problem first and an AI problem second**. Documents are ingested into an immutable Bronze layer, refined into typed Silver datasets, promoted into AI-ready Gold datasets, embedded, indexed into a vector store that is treated as a *derived cache*, and served through a retrieval-augmented generation (RAG) API that returns answers with verifiable citations.

The distinguishing property of AKL is that the **Lakehouse is the system of record and every downstream artefact (chunks, embeddings, vector indexes, BM25 indexes) is reproducible from it**. If Qdrant is deleted, it is rebuilt from Gold. If the embedding model changes, a new embedding version is generated from the same chunk dataset and the old version is retired. If a chunking bug is discovered, chunks are regenerated from Silver without re-fetching a single source document.

## 0.2 Problem Statement

Naive RAG implementations couple the vector database to the ingestion script. They load documents, chunk them, embed them, and push vectors in one process. This creates six structural failures that AKL is designed to eliminate:

| # | Failure in naive RAG | Consequence | AKL response |
|---|---|---|---|
| 1 | No immutable raw layer | Cannot re-parse after parser bug fix; cannot audit what was ingested | Bronze layer is append-only and content-addressed |
| 2 | Full re-embedding on every run | Cost and latency scale with corpus size, not with change volume | Chunk-level checksums + embedding cache; only changed chunks re-embed |
| 3 | Vector DB is system of record | Data loss on vector DB failure; no lineage; no reproducibility | Vector DB is derived state, rebuilt from Gold on demand |
| 4 | No versioning of chunks | Citations point to text that no longer exists | Chunk versioning with stable `chunk_key` and content hash |
| 5 | Dense-only retrieval | Poor recall on exact identifiers, error codes, function names | Hybrid BM25 + dense retrieval with cross-encoder reranking |
| 6 | No observability | Silent degradation of retrieval quality and freshness | Prometheus metrics, OpenTelemetry traces, MLflow evaluation runs, freshness SLAs |

## 0.3 Goals

| ID | Goal | Measurable statement |
|---|---|---|
| G1 | Multi-source ingestion | Ingest PDF, Markdown, HTML and GitHub repositories through a single connector interface |
| G2 | Lakehouse correctness | Bronze/Silver/Gold layers stored as Parquet on MinIO, queryable with DuckDB, with documented partitioning |
| G3 | Incremental everything | A re-run with zero source changes MUST perform zero parsing, zero chunking, zero embedding and zero vector upserts |
| G4 | Hybrid retrieval quality | Recall@10 ≥ 0.85 on the synthetic evaluation set; MRR ≥ 0.70 after reranking |
| G5 | Cited answers | 100% of generated answers carry ≥ 1 citation resolvable to a `chunk_id`, `document_id`, and source locator |
| G6 | Observability | Every pipeline stage and API request emits structured logs, metrics and traces with correlation IDs |
| G7 | Reproducibility | Deleting Qdrant and running the Qdrant Sync DAG restores an identical collection (same point count, same payloads) |
| G8 | Portability of design | The architecture document (Chapter 14) demonstrates the migration path to 1 PB without changing layer semantics |

## 0.4 Non-Goals

The following are explicitly **out of scope** for version 1.0.0:

- Training or fine-tuning embedding or generation models.
- Real-time (sub-second) ingestion. Freshness SLA is measured in minutes/hours, not seconds.
- Multi-tenant SaaS isolation across organisations. AKL is single-organisation, multi-team.
- Connectors beyond the four locked sources (Confluence, Slack, Notion, Jira are Future Enterprise Scale candidates via the plugin interface, not MVP).
- A rich front-end. The API is the product; a minimal OpenAPI UI is sufficient.
- Kubernetes deployment. Documented as a migration path (Chapter 11.14) but not implemented.
- OCR of scanned PDFs. Image-only pages are detected, flagged in metadata, and skipped.

## 0.5 Functional Requirements

| ID | Requirement | Chapter |
|---|---|---|
| FR-01 | System MUST accept document uploads (PDF, MD, HTML) via authenticated REST API | 10 |
| FR-02 | System MUST sync a configured GitHub repository (branch, path filter, file-type filter) on schedule and on demand | 3, 7 |
| FR-03 | Every ingested document MUST be written to Bronze with content checksum, source URI, fetch timestamp, connector version | 2, 3 |
| FR-04 | Documents failing validation MUST be routed to a quarantine prefix with a structured failure reason | 3 |
| FR-05 | Silver MUST contain parsed documents and a chunk dataset with heading path, token count, quality score, checksum | 2, 4 |
| FR-06 | Chunking MUST be hierarchical (heading → semantic → token) and MUST handle code blocks and tables as atomic units where they fit | 4 |
| FR-07 | Embeddings MUST be generated with `BAAI/bge-small-en-v1.5` (384 dimensions) and cached by `(chunk_checksum, model_id, model_version)` | 5 |
| FR-08 | Qdrant collection MUST be reconciled from Gold; stale points MUST be deleted, new points upserted | 5, 7 |
| FR-09 | Search API MUST support dense, sparse (BM25) and hybrid modes with metadata filters | 6, 10 |
| FR-10 | Chat API MUST return an answer, ordered citations, confidence score and retrieval trace; MUST support streaming | 6, 10 |
| FR-11 | System MUST refuse to answer (return `insufficient_evidence`) when top reranked score < configured threshold | 6 |
| FR-12 | System MUST support multi-turn conversation with summarised memory | 6 |
| FR-13 | Five Airflow DAGs MUST orchestrate ingestion, chunking, embedding, Qdrant sync and maintenance | 7 |
| FR-14 | All services MUST expose `/health` and `/metrics` | 8, 10 |
| FR-15 | RBAC MUST restrict retrieval to documents whose `security_level` the caller is authorised for | 9 |
| FR-16 | Document deletion MUST cascade: Silver/Gold tombstone → embedding tombstone → Qdrant delete → audit record | 9 |
| FR-17 | Admin API MUST expose reindex, reembed, quarantine listing, and pipeline status | 10 |
| FR-18 | Evaluation harness MUST log retrieval and RAG metrics to MLflow per run | 8, 12 |

## 0.6 Non-Functional Requirements

| ID | Category | Requirement (Current MVP) | Requirement (Enterprise Scale) |
|---|---|---|---|
| NFR-01 | Latency | Search p95 ≤ 800 ms for 10k chunks on 4 CPU cores | Search p95 ≤ 300 ms at 1B vectors with sharded Qdrant |
| NFR-02 | Latency | Chat first-token ≤ 2.5 s (excluding external LLM latency) | ≤ 1.5 s |
| NFR-03 | Throughput | Ingestion ≥ 50 documents/minute on laptop | ≥ 10k documents/minute on Spark cluster |
| NFR-04 | Throughput | Embedding ≥ 200 chunks/second CPU batched | ≥ 20k chunks/second on GPU fleet |
| NFR-05 | Freshness | GitHub changes reflected in retrieval within 60 minutes | Within 15 minutes |
| NFR-06 | Availability | Single-node; restart tolerant; no data loss on container restart | 99.9% API availability |
| NFR-07 | Durability | All layers persisted to MinIO volume; Postgres volume; Qdrant volume | Cross-region object replication |
| NFR-08 | Idempotency | Any DAG run re-executed with identical inputs produces identical outputs and zero duplicate rows | Same |
| NFR-09 | Observability | 100% of requests and tasks carry `request_id`/`run_id`; metrics scraped every 15 s | Same, plus long-term metric storage |
| NFR-10 | Security | JWT + API key auth; secrets from env/secret files only; TLS terminated at reverse proxy | mTLS between services; KMS-backed encryption |
| NFR-11 | Testability | ≥ 80% line coverage on `akl` package; integration tests run against Compose stack in CI | Same |
| NFR-12 | Portability | Runs on 16 GB RAM laptop, Docker Desktop, no GPU | Kubernetes, autoscaling |
| NFR-13 | Cost | $0 infrastructure for MVP (all local, all OSS) | Documented cost model per TB (Chapter 14) |

## 0.7 Success Metrics

| Metric | Target | Measurement method |
|---|---|---|
| `retrieval_recall_at_10` | ≥ 0.85 | Evaluation DAG against `gold/eval/qa_pairs` dataset, logged to MLflow |
| `retrieval_mrr` | ≥ 0.70 | Same |
| `answer_citation_coverage` | 1.00 | Fraction of answers with ≥1 valid citation |
| `answer_faithfulness` | ≥ 0.90 | LLM-as-judge or NLI check that answer sentences are entailed by cited chunks |
| `incremental_efficiency` | ≥ 0.95 | 1 − (chunks re-embedded / total chunks) on a no-change run; MUST be 1.0 |
| `data_freshness_minutes` | ≤ 60 | `now − max(gold.chunks.updated_at)` for GitHub source |
| `pipeline_success_rate` | ≥ 0.98 | Successful DAG runs / total DAG runs over 30 days |
| `search_latency_p95_ms` | ≤ 800 | Prometheus histogram `akl_search_latency_seconds` |
| `qdrant_gold_drift` | 0 | Points in Qdrant not present in Gold + Gold active chunks not in Qdrant, after sync |

## 0.8 Assumptions

| ID | Assumption |
|---|---|
| A1 | A generation LLM is available via an OpenAI-compatible HTTP endpoint (local, e.g. llama.cpp/Ollama server, or hosted). The locked stack does not name a generation model; AKL abstracts it behind `akl.rag.llm.LLMProvider` and ships an **extractive fallback** that answers by returning reranked passages verbatim when no LLM is configured. |
| A2 | Source documents are predominantly English. Language is detected and stored; non-English documents are ingested but flagged `language != "en"` and excluded from the default evaluation set. |
| A3 | The developer machine has ≥ 16 GB RAM, ≥ 4 CPU cores, ≥ 40 GB free disk, Docker Desktop or Docker Engine 24+. |
| A4 | GitHub API access uses a personal access token with read scope; rate limits (5,000 req/h authenticated) are respected via conditional requests (ETag) and the `git clone --depth 1` fallback. |
| A5 | Corpus size for MVP benchmarks: 500–2,000 documents, 20k–100k chunks. |
| A6 | Airflow runs with `LocalExecutor` in MVP. `CeleryExecutor` / `KubernetesExecutor` are Enterprise Scale. |

## 0.9 Constraints

| ID | Constraint | Implication |
|---|---|---|
| C1 | Zero-cost infrastructure | All services are open-source and run in Docker Compose on one host |
| C2 | Locked technology stack | No substitutions (e.g. no Elasticsearch for BM25; use `rank_bm25` + Parquet-backed index, with Qdrant sparse vectors as the Enterprise Scale path) |
| C3 | CPU-only embedding | Batch size, thread pinning and ONNX runtime are the primary levers; GPU is optional and auto-detected |
| C4 | No Kubernetes in MVP | Compose health checks and `depends_on` conditions provide startup ordering |
| C5 | Single Postgres instance shared by Airflow metadata, MLflow backend store and AKL metadata | Separate databases (`airflow`, `mlflow`, `akl`) within one instance; separate roles |
| C6 | Design must be implementable by an AI coding agent without clarification | Every module, schema, env var, metric and error code is fully specified in this document |

---

# Chapter 1 — System Architecture

## 1.1 Architectural Overview

AKL is composed of four planes:

1. **Ingestion plane** — Connectors, parsers, validators and the Bronze writer. Executed by Airflow tasks and by the FastAPI upload endpoint (which writes Bronze synchronously and triggers downstream DAGs asynchronously).
2. **Data plane (Lakehouse)** — MinIO object storage holding Parquet datasets in Bronze/Silver/Gold prefixes, queried and transformed by DuckDB. PostgreSQL holds the operational metadata catalogue (documents, versions, runs, cache, audit).
3. **AI plane** — Chunking engine, embedding pipeline, Qdrant vector store, BM25 sparse index, cross-encoder reranker, RAG engine, LLM provider abstraction, MLflow experiment tracking.
4. **Serving & control plane** — FastAPI gateway, Airflow scheduler/webserver, Prometheus, Grafana, OpenTelemetry collector.

## 1.2 Complete Architecture Diagram

```mermaid
flowchart TB
    subgraph Clients
        U[Engineer / Internal Tool]
        AD[Admin]
    end

    subgraph Serving["Serving Plane (FastAPI :8000)"]
        GW[API Gateway<br/>auth, rate limit, request_id]
        QP[Query Processor<br/>normalize, intent, entities, filters]
        HR[Hybrid Retrieval Engine<br/>BM25 + Dense + RRF]
        RR[Cross-Encoder Reranker]
        CB[Context Builder<br/>dedupe, order, budget]
        LLM[LLM Provider<br/>OpenAI-compatible / extractive fallback]
        CIT[Citation Engine]
        MEM[Conversation Memory]
    end

    subgraph AIState["AI Plane — Derived State"]
        QD[(Qdrant<br/>kb_chunks_v1<br/>384-d cosine, HNSW)]
        BM[(BM25 Index<br/>serialized to MinIO gold/indexes)]
        EMB[Embedding Service<br/>bge-small-en-v1.5, ONNX, batched]
        CACHE[(Embedding Cache<br/>Postgres + Parquet)]
    end

    subgraph Lakehouse["Data Plane — Lakehouse (MinIO :9000 + DuckDB)"]
        BR[(Bronze<br/>raw bytes + manifest.parquet)]
        SV[(Silver<br/>documents.parquet<br/>chunks.parquet)]
        GD[(Gold<br/>chunk_embeddings.parquet<br/>retrieval_units.parquet<br/>eval datasets)]
        QR[(Quarantine)]
    end

    subgraph Meta["Metadata"]
        PG[(PostgreSQL<br/>akl / airflow / mlflow DBs)]
        ML[MLflow :5000<br/>runs, eval metrics, model registry]
    end

    subgraph Orchestration["Airflow (:8080)"]
        D1[akl_ingestion]
        D2[akl_chunking]
        D3[akl_embedding]
        D4[akl_qdrant_sync]
        D5[akl_maintenance]
    end

    subgraph Sources
        GH[GitHub Repos]
        PDF[PDF Uploads]
        MD[Markdown]
        HTML[HTML Pages]
    end

    subgraph Obs["Observability"]
        PR[Prometheus :9090]
        GR[Grafana :3000]
        OT[OTel Collector :4317]
    end

    U --> GW --> QP --> HR
    HR --> QD
    HR --> BM
    HR --> RR --> CB --> LLM --> CIT --> GW
    QP --> EMB
    MEM <--> GW
    AD --> GW

    GH --> D1
    PDF --> GW
    MD --> D1
    HTML --> D1
    GW -- upload --> BR
    D1 --> BR
    D1 --> QR
    D1 --> SV
    D2 --> SV
    D3 --> GD
    D3 <--> CACHE
    D3 --> EMB
    D4 --> QD
    D4 --> BM
    D5 --> BR
    D5 --> SV
    D5 --> GD

    D1 --> PG
    D2 --> PG
    D3 --> PG
    D4 --> PG
    D3 --> ML
    GW --> PG
    HR -. metadata filters .-> PG

    GW --> PR
    D1 --> PR
    EMB --> PR
    PR --> GR
    GW --> OT
    D1 --> OT
```

## 1.3 Component Responsibilities

| Component | Responsibility | Owns state? | Failure impact |
|---|---|---|---|
| FastAPI Gateway | Authentication, authorization, rate limiting, request ID generation, routing, OpenAPI, streaming | No (stateless) | API unavailable; pipelines unaffected |
| Query Processor | Normalisation, spell correction, intent classification, entity extraction, metadata filter inference | No | Degraded retrieval quality |
| Hybrid Retrieval Engine | Parallel BM25 + dense retrieval, Reciprocal Rank Fusion, metadata filtering, security filtering | No | Search unavailable |
| Cross-Encoder Reranker | Re-scores top-N fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` | No | Falls back to fused scores (degraded) |
| Context Builder | Deduplication, ordering, token budget enforcement | No | — |
| LLM Provider | Generation via configurable backend; extractive fallback | No | Falls back to extractive answers |
| Citation Engine | Maps answer spans to chunk IDs and source locators | No | — |
| Conversation Memory | Stores turns in Postgres; summarises after N turns | Yes (Postgres) | Multi-turn context lost |
| Embedding Service | Loads BGE-small (ONNX), batches, normalises, emits vectors | No (model weights cached on volume) | Embedding DAG and query embedding fail |
| Embedding Cache | Lookup `(chunk_checksum, model_id, model_version) → vector` | Yes (Postgres index + Parquet payload) | Full re-embedding (slow, not incorrect) |
| Qdrant | ANN search over chunk vectors with payload filtering | Yes, **derived** | Dense retrieval unavailable; rebuildable from Gold |
| BM25 Index | Sparse lexical retrieval | Yes, **derived** | Sparse retrieval unavailable; rebuildable from Gold |
| MinIO | Object storage for all Lakehouse layers, indexes, MLflow artefacts, quarantine | Yes, **system of record** | Total outage; all reads/writes fail |
| DuckDB | Embedded query/transform engine over Parquet via S3 API; runs inside Airflow tasks and API process | No (in-process) | — |
| PostgreSQL | Operational metadata catalogue, auth, audit, conversations, cache index, Airflow & MLflow backends | Yes | Pipelines and auth fail |
| Airflow | Scheduling, retries, dependencies, backfills for the 5 DAGs | Yes (Airflow metadata DB) | No new data processed; serving unaffected |
| MLflow | Experiment tracking for embedding runs and evaluation; model registry for embedding/reranker versions | Yes (Postgres + MinIO artefacts) | Loss of experiment history only |
| Prometheus | Metric scraping and alert rule evaluation | Yes (TSDB volume) | Loss of monitoring |
| Grafana | Dashboards provisioned from repo | Yes (dashboards as code) | Loss of visualisation |
| OTel Collector | Receives traces from API and tasks, exports to Grafana Tempo (optional) or logs | No | Loss of tracing |

## 1.4 Service Communication

| Caller | Callee | Protocol | Port | Auth | Notes |
|---|---|---|---|---|---|
| Client | FastAPI | HTTPS (TLS at reverse proxy) / HTTP in Compose | 8000 | JWT or API key | REST + SSE streaming |
| FastAPI | Qdrant | gRPC (preferred) / HTTP | 6334 / 6333 | API key (optional) | `qdrant-client` async |
| FastAPI | PostgreSQL | TCP (asyncpg) | 5432 | role `akl_api` | Connection pool 5–20 |
| FastAPI | MinIO | S3 HTTP | 9000 | access/secret key | Upload → Bronze |
| FastAPI | Airflow REST API | HTTP | 8080 | basic auth (service user) | Trigger DAG runs after upload |
| FastAPI | LLM endpoint | HTTP | configurable | bearer | OpenAI-compatible `/v1/chat/completions` |
| Airflow tasks | MinIO | S3 HTTP via DuckDB `httpfs` and `boto3` | 9000 | access/secret | All Lakehouse IO |
| Airflow tasks | PostgreSQL | TCP (psycopg) | 5432 | role `akl_pipeline` | Metadata upserts |
| Airflow tasks | Qdrant | gRPC | 6334 | — | Batch upsert/delete |
| Airflow tasks | MLflow | HTTP | 5000 | — | Log metrics/artefacts |
| Prometheus | all services | HTTP `/metrics` | various | — | 15 s scrape |
| Services | OTel Collector | gRPC OTLP | 4317 | — | Traces |

All internal calls MUST propagate `X-Request-ID` (API) or `run_id`/`task_id` (Airflow) as `correlation_id` in logs and as a span attribute.

## 1.5 Data Flow

```mermaid
flowchart LR
    S[Source bytes] -->|connector.fetch| B1[Bronze object<br/>raw/<source>/<doc_hash>.bin]
    S -->|connector.describe| B2[Bronze manifest row]
    B1 & B2 -->|parser| S1[Silver documents.parquet]
    S1 -->|chunker| S2[Silver chunks.parquet]
    S2 -->|quality gate| S3{quality ≥ threshold?}
    S3 -->|yes| G1[Gold retrieval_units.parquet]
    S3 -->|no| Q[Quarantine / low_quality flag]
    G1 -->|embedder + cache| G2[Gold chunk_embeddings.parquet]
    G2 -->|reconciler| QD[(Qdrant points)]
    G1 -->|bm25 builder| BM[(BM25 index)]
    QD & BM -->|hybrid retrieval| A[Answer + citations]
```

Each arrow is an **idempotent, incremental** transformation keyed on a checksum:

| Transformation | Idempotency key | Skip condition |
|---|---|---|
| Fetch → Bronze | `content_sha256` | Object with same sha exists in Bronze |
| Bronze → Silver documents | `(content_sha256, parser_version)` | Silver row exists with same key |
| Silver documents → Silver chunks | `(document_version_id, chunker_version, chunk_config_hash)` | Chunks exist for same key |
| Silver chunks → Gold embeddings | `(chunk_checksum, model_id, model_version)` | Embedding cache hit |
| Gold → Qdrant | `(chunk_id, embedding_version)` | Point exists with same `payload.embedding_version` |

## 1.6 Request Lifecycle (Chat)

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as FastAPI Gateway
    participant QP as Query Processor
    participant E as Embedding Service
    participant Q as Qdrant
    participant B as BM25 Index
    participant RR as Reranker
    participant CB as Context Builder
    participant L as LLM Provider
    participant CI as Citation Engine
    participant PG as Postgres

    C->>GW: POST /v1/chat (query, conversation_id, filters)
    GW->>GW: authn (JWT), authz (allowed security_levels), request_id
    GW->>PG: load conversation memory (summary + last k turns)
    GW->>QP: process(query, memory)
    QP->>QP: normalize, spell-correct, intent, entities, infer filters
    par dense
        QP->>E: embed(rewritten_query)
        E-->>QP: vector[384]
        QP->>Q: search(vector, filter: security_level ∈ allowed ∧ inferred, limit=50)
        Q-->>QP: dense candidates
    and sparse
        QP->>B: search(tokens, limit=50)
        B-->>QP: sparse candidates
    end
    QP->>QP: Reciprocal Rank Fusion → top 40
    QP->>RR: rerank(query, 40 candidates)
    RR-->>QP: scored candidates
    alt top score < AKL_RAG_MIN_CONFIDENCE
        QP-->>GW: insufficient_evidence
        GW-->>C: 200 {answer: null, reason: insufficient_evidence, candidates}
    else
        QP->>CB: build(top 8, budget=AKL_RAG_CONTEXT_TOKENS)
        CB-->>L: prompt(system, context blocks with [n] markers, history, query)
        L-->>GW: token stream
        GW-->>C: SSE stream (tokens)
        L-->>CI: full answer
        CI->>CI: extract [n] markers, map to chunk_id, validate
        CI-->>GW: citations[]
        GW->>PG: persist turn, citations, retrieval trace
        GW-->>C: SSE final event {citations, confidence, trace_id}
    end
```

## 1.7 Storage Architecture

| Store | Technology | Content | Format | Consistency |
|---|---|---|---|---|
| Object store | MinIO | Bronze raw bytes, Bronze manifest, Silver Parquet, Gold Parquet, BM25 index artefact, MLflow artefacts, quarantine | Binary + Parquet (ZSTD) | Read-after-write per object |
| Query engine | DuckDB | Ephemeral; reads Parquet over `s3://` via `httpfs`; writes Parquet with `COPY ... TO` | In-process | Per-task |
| Metadata DB | PostgreSQL 16 | Catalogue tables (Appendix A) | Relational | ACID |
| Vector store | Qdrant | Collection `kb_chunks_v1` | HNSW + payload | Eventually consistent with Gold; reconciled by DAG |
| Cache | Postgres `embedding_cache` table + Gold Parquet | Vector by checksum | Relational index + Parquet | Derived |

**Why both Parquet and Postgres for metadata?** Parquet is the analytical, append-friendly, columnar record of every version (cheap to scan, cheap to store, partition-prunable). Postgres holds the *current-state index* that needs point lookups and transactional updates (document status, latest version pointer, cache hits, auth). Postgres is rebuildable from Parquet; the reverse is not required.

## 1.8 Rationale for Core Technology Choices

### 1.8.1 Why Lakehouse over Traditional Warehouse

| Dimension | Warehouse (e.g. row-store RDBMS as document store) | Lakehouse (Parquet + object storage + query engine) |
|---|---|---|
| Storage cost | High (block storage, replication) | Low (object storage, columnar compression 5–10×) |
| Unstructured data | Poor (BLOBs) | Native (raw bytes alongside tabular metadata) |
| Schema evolution | Migrations with locks | Additive columns per file; reader merges schemas |
| Compute/storage coupling | Coupled | Decoupled; scale DuckDB → Spark without moving data |
| Immutability | Hard to enforce | Natural (write-once objects, versioned prefixes) |
| Reprocessing | Expensive | Cheap; re-derive Silver from Bronze anytime |
| Vendor lock-in | High | Low; open formats |

The RAG workload is append-heavy, scan-heavy, and rarely updates in place. That is the exact access pattern a Lakehouse optimises for.

### 1.8.2 Why Parquet

- Columnar layout: chunk-level analytics (token distributions, quality scores) scan only needed columns.
- Row-group statistics (min/max, null count) enable predicate pushdown on `document_id`, `ingest_date`, `source_type`.
- Dictionary encoding compresses repeated strings (`source_type`, `language`, `heading_path`) heavily.
- Nested types: `heading_path` as `LIST<VARCHAR>`, `metadata` as `MAP<VARCHAR,VARCHAR>`, embeddings as `LIST<FLOAT>` (fixed-size 384).
- Universal reader support: DuckDB, Spark, Polars, Arrow, Trino — the Enterprise Scale migration path requires no format conversion.

### 1.8.3 Why DuckDB

- In-process: no server to operate; runs inside Airflow task and API process.
- Native `httpfs` extension reads/writes `s3://` (MinIO) with predicate pushdown and Hive partition discovery.
- Vectorised execution handles 10–100 GB Parquet on a laptop.
- SQL is the transformation language → transformations are portable to Spark SQL / Trino at scale.
- `COPY (SELECT …) TO 's3://…' (FORMAT PARQUET, PARTITION_BY (…), COMPRESSION ZSTD)` performs partitioned writes directly.

Limitation acknowledged: single-node. Chapter 14 defines the point (≈1 TB active Silver) at which transformations migrate to Spark while keeping identical SQL semantics.

### 1.8.4 Why MinIO

- S3-compatible API → every line of storage code (`boto3`, DuckDB `httpfs`, MLflow artefact store) works unchanged against AWS S3, GCS (S3 interop) or Ceph at Enterprise Scale.
- Bucket versioning and object lock provide Bronze immutability enforcement.
- Lifecycle rules provide retention (Chapter 9.14).
- Runs in one container with one volume.

### 1.8.5 Why Qdrant

- Payload indexing enables **filtered ANN** (security level, source type, repo, date) without post-filtering recall loss.
- Native sparse vectors provide the Enterprise Scale path to move BM25 into Qdrant (hybrid in one engine).
- gRPC API, async Python client, snapshots for backup, collection aliases for zero-downtime reindex (`kb_chunks` alias → `kb_chunks_v1`/`kb_chunks_v2`).
- Horizontal sharding and replication built in (Chapter 14).
- Quantisation (scalar/product) reduces memory 4–32× at scale.

### 1.8.6 Why PostgreSQL

- Single engine for three metadata consumers (Airflow, MLflow, AKL) reduces operational surface in MVP.
- Transactional current-state updates (document status transitions, version pointers).
- `pgcrypto`/row-level policies available for RBAC at Enterprise Scale.
- JSONB for flexible per-source metadata without schema churn.

## 1.9 Architecture Decision Records

Each ADR follows: Context → Decision → Alternatives → Consequences.

### ADR-001 — Vector database is derived state

- **Context**: Vector stores are operationally fragile and version-coupled to embedding models.
- **Decision**: Gold `chunk_embeddings.parquet` is the durable record; Qdrant is a rebuildable index. A reconciler DAG computes the diff between Gold and Qdrant and applies it.
- **Alternatives**: Qdrant as source of truth (rejected: no lineage, no cheap analytics, model migration requires re-embedding from unknown state).
- **Consequences**: Extra storage (~1.5 KB/chunk for 384 float32 in Parquet, ZSTD ≈ 1.2 KB). Reconciliation adds a DAG. Gains full reproducibility (G7).

### ADR-002 — Content-addressed Bronze

- **Context**: Sources are re-fetched on schedule; identical content must not create new work.
- **Decision**: Bronze object key = `bronze/raw/source_type=<t>/sha256=<hash>.<ext>`; manifest is an append-only Parquet dataset partitioned by `ingest_date`. Documents are identified by `document_id = uuid5(namespace, canonical_source_uri)`; versions by `content_sha256`.
- **Alternatives**: Key by source URI + timestamp (rejected: duplicates identical content; loses dedup across mirrors).
- **Consequences**: Same content under two URIs stored once; manifest maps both URIs to one sha.

### ADR-003 — Chunk identity has two keys

- **Context**: Citations need stability across re-chunking; embeddings need exact content identity.
- **Decision**: `chunk_key = sha1(document_id + heading_path + ordinal_within_heading)` (stable across small edits, used for version lineage) and `chunk_checksum = sha256(normalized_text)` (content identity, used for embedding cache). `chunk_id = uuid5(namespace, document_id + ":" + chunk_key + ":" + chunk_checksum)`.
- **Alternatives**: Single content hash (rejected: cannot express "chunk X was updated"); positional index (rejected: shifts on any insertion).
- **Consequences**: Incremental update algorithm (Chapter 4.12) can classify chunks as unchanged/modified/added/removed.

### ADR-004 — Embedding cache keyed by content, not by chunk

- **Decision**: Cache key `(chunk_checksum, model_id, model_version, normalize=true)`. Identical text across documents (license boilerplate, repeated headers) embeds once.
- **Consequences**: Cache hit rate on real corpora typically 5–20% beyond incremental skip; enables cheap re-chunking when boundaries move but text is preserved.

### ADR-005 — Hybrid retrieval with Reciprocal Rank Fusion then cross-encoder rerank

- **Decision**: Retrieve top-50 dense + top-50 sparse, fuse with RRF (k=60), rerank top-40 with cross-encoder, return top-8 to context.
- **Alternatives**: Weighted score fusion (rejected: dense cosine and BM25 scores are on incomparable scales; RRF is scale-free); dense-only (rejected: fails on identifiers).
- **Consequences**: Reranker adds ~150–300 ms CPU for 40 pairs; acceptable within NFR-01. Reranker is optional per request (`rerank=false`).

### ADR-006 — DuckDB as MVP transformation engine, SQL as the contract

- **Decision**: All Silver/Gold transformations are expressed as SQL files under `akl/lakehouse/sql/`, executed by DuckDB. Python only orchestrates and handles non-SQL steps (parsing, chunking, embedding).
- **Consequences**: Spark/Trino migration is a dialect change, not a rewrite.

### ADR-007 — Airflow `LocalExecutor` in MVP

- **Decision**: One scheduler container, one worker container (LocalExecutor runs tasks in scheduler; a dedicated "worker" container runs `airflow celery worker` only when `AKL_AIRFLOW_EXECUTOR=CeleryExecutor`). MVP default is LocalExecutor with a separate container reserved so the Compose topology does not change on upgrade.
- **Consequences**: Parallelism bounded by scheduler `parallelism=8`.

### ADR-008 — BM25 index as a serialised artefact, not a service

- **Context**: Stack forbids Elasticsearch/OpenSearch.
- **Decision**: Build BM25 (`rank_bm25.BM25Okapi`, tokenizer = lowercase + Unicode word boundary + code-identifier splitting) from Gold `retrieval_units`, serialise with `pickle` + metadata JSON to `gold/indexes/bm25/version=<v>/`, load into API process memory at startup and on `/admin/reload-index`.
- **Alternatives**: Qdrant sparse vectors (chosen for Enterprise Scale, Chapter 14), DuckDB FTS extension (viable alternative; documented but not default due to BM25 parameter control).
- **Consequences**: Memory ≈ 300–600 MB for 100k chunks; rebuild time ≈ 30 s; acceptable.

### ADR-009 — Security filtering is applied inside retrieval, not after

- **Decision**: `security_level` and `allowed_groups` are Qdrant payload fields with indexes; the API injects a filter derived from the caller's principal into every Qdrant query and BM25 candidate filter.
- **Consequences**: No recall loss from post-filtering; no leakage of restricted chunk text into reranking.

### ADR-010 — Extractive fallback when no LLM is configured

- **Decision**: If `AKL_LLM_PROVIDER=none`, Chat API returns the top-3 reranked passages as the answer with citations and `mode: extractive`. All evaluation of retrieval quality is independent of the generation model.
- **Consequences**: The platform is fully demonstrable with zero external dependencies.

---

# Chapter 2 — Lakehouse Design

## 2.1 Layer Semantics

| Layer | Purpose | Mutability | Producer | Consumer | Format |
|---|---|---|---|---|---|
| Bronze | Faithful copy of source bytes plus fetch metadata | Append-only; objects immutable; manifest append-only | Connectors (Airflow `akl_ingestion`, API upload) | Parsers | Raw bytes + Parquet manifest |
| Silver | Parsed, cleaned, normalised, deduplicated documents and chunks | Append-only versions; current view via `is_current` flag and Postgres pointer | Parsers, Chunking engine | Embedding pipeline, analytics, eval | Parquet |
| Gold | AI-ready retrieval units, embeddings, indexes, evaluation datasets | Append-only versions; compaction rewrites partitions atomically | Embedding pipeline, index builders, eval generator | Qdrant reconciler, BM25 builder, RAG evaluation | Parquet + serialised indexes |
| Quarantine | Inputs that failed validation or parsing | Append-only | Validators | Admin API, Maintenance DAG | Raw bytes + Parquet reasons |

## 2.2 Bucket and Prefix Hierarchy

MinIO bucket: `akl-lakehouse` (single bucket, prefix-per-layer; ADR rationale: one lifecycle policy set, one credential in MVP; Enterprise Scale splits into buckets per layer for IAM isolation).

```
s3://akl-lakehouse/
├── bronze/
│   ├── raw/
│   │   └── source_type={pdf|markdown|html|github}/
│   │       └── sha256=<64 hex>.<ext>                  # immutable raw bytes
│   ├── manifest/
│   │   └── ingest_date=YYYY-MM-DD/
│   │       └── part-<run_id>-<n>.parquet              # one row per fetched object
│   └── github_snapshots/
│       └── repo=<owner>__<name>/
│           └── commit=<sha>/
│               └── tree.parquet                       # file listing at commit
├── silver/
│   ├── documents/
│   │   └── source_type=<t>/
│   │       └── ingest_date=YYYY-MM-DD/
│   │           └── part-<run_id>-<n>.parquet
│   ├── chunks/
│   │   └── source_type=<t>/
│   │       └── ingest_date=YYYY-MM-DD/
│   │           └── part-<run_id>-<n>.parquet
│   └── dedup_ledger/
│       └── part-*.parquet                             # fingerprint → canonical document_id
├── gold/
│   ├── retrieval_units/
│   │   └── source_type=<t>/
│   │       └── security_level=<internal|public|restricted>/
│   │           └── part-*.parquet
│   ├── chunk_embeddings/
│   │   └── embedding_version=<model_id>__<model_version>__<dim>/
│   │       └── source_type=<t>/
│   │           └── part-*.parquet
│   ├── indexes/
│   │   └── bm25/
│   │       └── version=<gold_snapshot_id>/
│   │           ├── index.pkl
│   │           └── meta.json
│   ├── eval/
│   │   ├── qa_pairs/version=<v>/part-*.parquet
│   │   └── results/run_date=YYYY-MM-DD/part-*.parquet
│   └── stats/
│       └── snapshot_date=YYYY-MM-DD/part-*.parquet    # corpus statistics
├── quarantine/
│   └── ingest_date=YYYY-MM-DD/
│       ├── objects/sha256=<hash>.<ext>
│       └── reasons/part-*.parquet
└── mlflow/
    └── artifacts/...                                  # MLflow artefact root
```

## 2.3 Partitioning Strategy

| Dataset | Partition columns | Rationale | Anti-pattern avoided |
|---|---|---|---|
| `bronze/manifest` | `ingest_date` | Incremental DAG reads only today's/yesterday's partitions | Partition by `document_id` (millions of tiny files) |
| `silver/documents` | `source_type`, `ingest_date` | Parsers differ per source; incremental by date | — |
| `silver/chunks` | `source_type`, `ingest_date` | Embedding DAG selects new chunks by date; source-type analytics | Partition by `document_id` |
| `gold/retrieval_units` | `source_type`, `security_level` | Retrieval filters are dominated by source and security → pruning at scan | Partition by date (retrieval doesn't filter by date) |
| `gold/chunk_embeddings` | `embedding_version`, `source_type` | Model migration = new partition; old partition dropped after cutover | Mixing versions in one partition |

**File sizing target**: 128–512 MB per Parquet file after compaction (Chapter 2.8). Incremental writes may produce small files; the Maintenance DAG compacts them.

**Row group size**: 122,880 rows (DuckDB default) for chunks; 10,000 rows for `chunk_embeddings` (wide rows: 384 floats) to keep row groups ≈ 15 MB.

### 2.3.1 Partition Pruning and Predicate Pushdown

DuckDB `read_parquet('s3://akl-lakehouse/silver/chunks/*/*/*.parquet', hive_partitioning=true)` exposes `source_type` and `ingest_date` as virtual columns. A query with `WHERE ingest_date = '2026-09-04'` lists only that prefix (pruning). Within files, row-group statistics for `document_id` (sorted on write via `ORDER BY document_id, chunk_index`) allow skipping row groups whose `[min,max]` range excludes the predicate (pushdown). Implementers MUST sort on the primary lookup column before `COPY … TO` to make pushdown effective.

## 2.4 Bronze Layer

### 2.4.1 Immutability Guarantees

- Bucket versioning enabled; raw object keys are content-addressed, so a re-upload of identical bytes is a no-op.
- Object Lock in governance mode (retention = `AKL_BRONZE_RETENTION_DAYS`) when `AKL_BRONZE_OBJECT_LOCK=true`; disabled by default in MVP because it requires bucket creation with lock enabled.
- Application code has no delete path to `bronze/raw/*` except the Maintenance DAG `bronze_retention` task, which only acts on objects whose sha is absent from every current Silver row and older than retention.

### 2.4.2 Bronze Manifest Schema (`bronze/manifest`)

| Column | Type | Description |
|---|---|---|
| `manifest_id` | UUID | Row identity |
| `document_id` | UUID | `uuid5(AKL_NS, canonical_source_uri)` |
| `content_sha256` | VARCHAR(64) | Hash of raw bytes |
| `source_type` | VARCHAR | `pdf`, `markdown`, `html`, `github` |
| `source_uri` | VARCHAR | Original locator (URL, path, `github://owner/repo@sha/path`) |
| `canonical_source_uri` | VARCHAR | Normalised locator (lowercased host, stripped query/fragment, commit replaced by branch for GitHub) |
| `object_key` | VARCHAR | Bronze raw key |
| `size_bytes` | BIGINT | |
| `mime_type` | VARCHAR | Detected via `python-magic` |
| `fetched_at` | TIMESTAMP WITH TIME ZONE | |
| `connector_name` | VARCHAR | |
| `connector_version` | VARCHAR | Semver |
| `run_id` | VARCHAR | Airflow run_id or `api:<request_id>` |
| `source_metadata` | MAP<VARCHAR,VARCHAR> | ETag, last-modified, repo, branch, commit, path, uploader |
| `ingest_date` | DATE | Partition |

### 2.4.3 GitHub Snapshot Table (`bronze/github_snapshots`)

| Column | Type |
|---|---|
| `repo` | VARCHAR |
| `commit_sha` | VARCHAR(40) |
| `path` | VARCHAR |
| `blob_sha` | VARCHAR(40) |
| `size_bytes` | BIGINT |
| `mode` | VARCHAR |
| `snapshot_at` | TIMESTAMPTZ |

Purpose: diffing two snapshots yields added/modified/deleted paths without re-downloading blobs (Chapter 3.5.4).

## 2.5 Silver Layer

### 2.5.1 `silver/documents` Schema

| Column | Type | Description |
|---|---|---|
| `document_version_id` | UUID | `uuid5(AKL_NS, document_id + content_sha256 + parser_version)` |
| `document_id` | UUID | Stable across versions |
| `content_sha256` | VARCHAR(64) | Links to Bronze |
| `source_type` | VARCHAR | |
| `canonical_source_uri` | VARCHAR | |
| `title` | VARCHAR | Extracted or inferred |
| `language` | VARCHAR(8) | ISO 639-1, via `lingua`/`langdetect` |
| `text` | VARCHAR | Cleaned full text (Markdown-normalised representation) |
| `text_sha256` | VARCHAR(64) | Hash of cleaned text (parser-output identity) |
| `structure` | JSON | Heading tree: `[{level, text, start_char, end_char, children}]` |
| `tables` | JSON | Extracted tables: `[{id, page/section, markdown, n_rows, n_cols}]` |
| `code_blocks` | JSON | `[{id, language, start_char, end_char}]` |
| `images` | JSON | `[{id, page, alt, width, height, caption}]` |
| `page_map` | JSON | PDF: `[{page, start_char, end_char}]` |
| `word_count` | INTEGER | |
| `char_count` | INTEGER | |
| `quality_score` | FLOAT | 0–1 (Chapter 3.11) |
| `quality_flags` | LIST<VARCHAR> | `low_text_density`, `image_only_pages`, `encoding_issues`, `boilerplate_heavy` |
| `fingerprint_simhash` | UBIGINT | 64-bit SimHash for near-dup detection |
| `is_duplicate_of` | UUID | Canonical `document_id` if near-duplicate |
| `security_level` | VARCHAR | `public`, `internal`, `restricted` |
| `allowed_groups` | LIST<VARCHAR> | RBAC groups |
| `metadata` | MAP<VARCHAR,VARCHAR> | Source-specific |
| `parser_name` | VARCHAR | |
| `parser_version` | VARCHAR | |
| `parsed_at` | TIMESTAMPTZ | |
| `is_current` | BOOLEAN | Latest version flag (maintained by DAG; authoritative pointer is Postgres) |
| `is_deleted` | BOOLEAN | Tombstone |
| `ingest_date` | DATE | Partition |

### 2.5.2 `silver/chunks` Schema

| Column | Type | Description |
|---|---|---|
| `chunk_id` | UUID | See ADR-003 |
| `chunk_key` | VARCHAR(40) | Stable lineage key |
| `chunk_checksum` | VARCHAR(64) | sha256 of normalised text |
| `document_id` | UUID | |
| `document_version_id` | UUID | |
| `chunk_index` | INTEGER | Ordinal within document version |
| `chunk_type` | VARCHAR | `prose`, `code`, `table`, `heading_only`, `list`, `mixed` |
| `heading_path` | LIST<VARCHAR> | e.g. `["Installation","Docker","Volumes"]` |
| `heading_level` | SMALLINT | Depth of innermost heading |
| `text` | VARCHAR | Chunk text (with context prefix excluded) |
| `context_prefix` | VARCHAR | Title + heading path rendered as breadcrumb, prepended at embedding time |
| `start_char` | INTEGER | Offset in `silver.documents.text` |
| `end_char` | INTEGER | |
| `page_start` | INTEGER | PDF only |
| `page_end` | INTEGER | |
| `token_count` | INTEGER | `bge` tokenizer count |
| `overlap_prev_tokens` | INTEGER | |
| `language` | VARCHAR(8) | |
| `code_language` | VARCHAR | For `code` chunks |
| `quality_score` | FLOAT | Chapter 4.9 |
| `quality_flags` | LIST<VARCHAR> | |
| `prev_chunk_id` | UUID | Sequence linkage |
| `next_chunk_id` | UUID | |
| `parent_chunk_id` | UUID | Hierarchical parent (section-level) |
| `chunker_version` | VARCHAR | |
| `chunk_config_hash` | VARCHAR(16) | Hash of chunk config YAML |
| `security_level` | VARCHAR | Denormalised from document |
| `allowed_groups` | LIST<VARCHAR> | Denormalised |
| `source_type` | VARCHAR | Partition |
| `created_at` | TIMESTAMPTZ | |
| `is_current` | BOOLEAN | |
| `is_deleted` | BOOLEAN | |
| `ingest_date` | DATE | Partition |

### 2.5.3 Deduplication Ledger (`silver/dedup_ledger`)

| Column | Type |
|---|---|
| `fingerprint_simhash` | UBIGINT |
| `canonical_document_id` | UUID |
| `duplicate_document_id` | UUID |
| `hamming_distance` | SMALLINT |
| `decided_at` | TIMESTAMPTZ |

Exact duplicates are eliminated at Bronze (same sha). Near-duplicates (Hamming distance ≤ 3 on 64-bit SimHash over shingled cleaned text) are marked in Silver; only the canonical document proceeds to Gold. The canonical is the earliest ingested unless the later one has a higher `quality_score` by ≥ 0.1.

## 2.6 Gold Layer

### 2.6.1 `gold/retrieval_units` Schema

A projection of current, non-deleted, non-duplicate, quality-passing Silver chunks joined with document-level fields needed at query time. This is the **exact payload contract** for Qdrant and BM25.

| Column | Type | In Qdrant payload? | Qdrant payload index? |
|---|---|---|---|
| `chunk_id` | UUID | point id | — |
| `chunk_key` | VARCHAR | yes | no |
| `chunk_checksum` | VARCHAR | yes | no |
| `document_id` | UUID | yes | keyword |
| `document_version_id` | UUID | yes | no |
| `source_type` | VARCHAR | yes | keyword |
| `canonical_source_uri` | VARCHAR | yes | no |
| `title` | VARCHAR | yes | text (optional) |
| `heading_path` | LIST<VARCHAR> | yes | no |
| `heading_breadcrumb` | VARCHAR | yes | no |
| `chunk_type` | VARCHAR | yes | keyword |
| `code_language` | VARCHAR | yes | keyword |
| `text` | VARCHAR | yes | no |
| `token_count` | INTEGER | yes | integer |
| `page_start`,`page_end` | INTEGER | yes | no |
| `security_level` | VARCHAR | yes | keyword |
| `allowed_groups` | LIST<VARCHAR> | yes | keyword |
| `repo`,`branch`,`path` | VARCHAR | yes | keyword (repo) |
| `document_updated_at` | TIMESTAMPTZ | yes (epoch int) | integer |
| `quality_score` | FLOAT | yes | float |
| `gold_snapshot_id` | VARCHAR | yes | no |

### 2.6.2 `gold/chunk_embeddings` Schema

| Column | Type | Description |
|---|---|---|
| `chunk_id` | UUID | |
| `chunk_checksum` | VARCHAR(64) | Cache key component |
| `embedding_version` | VARCHAR | `bge-small-en-v1.5__1.5__384` |
| `model_id` | VARCHAR | |
| `model_version` | VARCHAR | |
| `dim` | SMALLINT | 384 |
| `vector` | FLOAT[384] | Fixed-size list, L2-normalised |
| `embedded_text_sha256` | VARCHAR(64) | sha of `context_prefix + "\n" + text` actually embedded |
| `embedded_at` | TIMESTAMPTZ | |
| `embedder_version` | VARCHAR | Code version |
| `mlflow_run_id` | VARCHAR | |
| `source_type` | VARCHAR | Partition |

### 2.6.3 Evaluation Datasets (`gold/eval/qa_pairs`)

| Column | Type |
|---|---|
| `qa_id` | UUID |
| `question` | VARCHAR |
| `expected_chunk_ids` | LIST<UUID> |
| `expected_document_id` | UUID |
| `reference_answer` | VARCHAR |
| `generation_method` | VARCHAR (`synthetic_llm`, `heading_question`, `manual`) |
| `difficulty` | VARCHAR |
| `version` | VARCHAR |

### 2.6.4 Corpus Statistics (`gold/stats`)

Daily snapshot: documents per source, chunks per source, token histogram buckets, quality score distribution, duplicate rate, embedding coverage, Qdrant drift. Feeds Grafana "Corpus Health" dashboard (Chapter 8.6).

## 2.7 Schema Evolution

Rules:

1. **Additive only** within a major layer version. New columns MUST be nullable. Readers use `union_by_name=true` in DuckDB.
2. **Renames and type changes** require a new dataset version prefix (`silver/chunks_v2/`) and a backfill task in the Maintenance DAG; the old prefix is retained until cutover and then archived.
3. Every dataset carries a `_schema_version` in the Parquet key-value metadata (`akl.schema_version=2.1.0`), written by the `ParquetWriter` wrapper.
4. Schemas are declared once in `akl/lakehouse/schemas/*.py` as `pyarrow.Schema` objects and enforced on write (`cast` with `safe=true`; failure → task error `AKL-E2101`).
5. A schema registry table `lakehouse_schema_versions` in Postgres records `(dataset, schema_version, first_written_at, pyarrow_schema_json)`.

## 2.8 Compaction Strategy

Incremental runs write one file per task per partition. After 30 daily runs, `silver/chunks/source_type=github/ingest_date=…` accumulates small files; more critically, `gold/retrieval_units` (not date-partitioned) accumulates appended version files.

Maintenance DAG task `compact_partitions`:

1. For each dataset/partition, list files; if `count > AKL_COMPACT_MIN_FILES (default 8)` or any file `< AKL_COMPACT_SMALL_FILE_MB (default 32)`:
2. DuckDB: `COPY (SELECT * FROM read_parquet(files) ORDER BY <sort key>) TO 's3://…/_compacting/<partition>/part-<run_id>.parquet'`.
3. Verify row count equality and checksum of sorted `chunk_id` list.
4. Atomic swap: copy compacted file to the partition path, then delete the original files (S3 has no atomic rename; the manifest table `lakehouse_files` in Postgres records the authoritative file list, and readers use it when `AKL_LAKEHOUSE_USE_FILE_MANIFEST=true`). MVP default reads by glob and tolerates the brief window where both old and new files exist by deduplicating on primary key in the reading SQL (`QUALIFY row_number() OVER (PARTITION BY chunk_id ORDER BY created_at DESC) = 1`).
5. Emit `akl_compaction_files_merged_total`, `akl_compaction_bytes_before/after`.

Enterprise Scale: Iceberg/Delta table formats provide atomic commits and hidden partitioning; Chapter 14 defines the migration.

## 2.9 Current-State Views

Because layers are append-only, "current" is a view. DuckDB view definitions live in `akl/lakehouse/sql/views/`:

- `v_current_documents`: latest `document_version_id` per `document_id` where `is_deleted=false`.
- `v_current_chunks`: chunks belonging to `v_current_documents` and matching `chunker_version = AKL_CHUNKER_VERSION` and `chunk_config_hash = current`.
- `v_gold_active_units`: `v_current_chunks` ∩ quality ≥ threshold ∩ not duplicate.
- `v_embedding_coverage`: `v_gold_active_units` LEFT JOIN `chunk_embeddings` for current `embedding_version`; rows with NULL vector are the embedding backlog.

Postgres mirrors the pointers (`documents.current_version_id`) for O(1) API lookups.

## 2.10 Lakehouse ER Diagram

```mermaid
erDiagram
    BRONZE_MANIFEST ||--o{ SILVER_DOCUMENTS : "content_sha256"
    SILVER_DOCUMENTS ||--o{ SILVER_CHUNKS : "document_version_id"
    SILVER_CHUNKS ||--o| GOLD_RETRIEVAL_UNITS : "chunk_id (current, quality-pass)"
    GOLD_RETRIEVAL_UNITS ||--o{ GOLD_CHUNK_EMBEDDINGS : "chunk_id + embedding_version"
    GOLD_RETRIEVAL_UNITS ||--o{ QDRANT_POINTS : "chunk_id = point_id"
    GOLD_RETRIEVAL_UNITS }o--|| BM25_INDEX : "doc position"
    SILVER_DOCUMENTS }o--o| DEDUP_LEDGER : "fingerprint"
    GOLD_EVAL_QA_PAIRS }o--o{ GOLD_RETRIEVAL_UNITS : "expected_chunk_ids"
```

---

# Chapter 3 — Document Ingestion Engine

## 3.1 Design Goals

- Uniform interface for heterogeneous sources.
- Separation of **fetch** (bytes + source metadata → Bronze) from **parse** (Bronze → Silver). Fetch failures never lose bytes already fetched; parse failures never require re-fetch.
- Every stage idempotent and incremental.
- Deterministic output: same bytes + same parser version → byte-identical Silver text.

## 3.2 Plugin Architecture

Connectors and parsers are discovered through Python entry points (`akl.connectors`, `akl.parsers`) declared in `pyproject.toml`, and additionally by an explicit registry in `akl/ingestion/registry.py` for deterministic ordering. Adding a source requires: a connector class, a parser class, a YAML config schema, tests, and a registry entry. No changes to DAGs.

```mermaid
classDiagram
    class BaseConnector {
        <<abstract>>
        +name: str
        +version: str
        +config: ConnectorConfig
        +discover(state: ConnectorState) Iterator~SourceItem~
        +fetch(item: SourceItem) FetchedObject
        +checkpoint(state: ConnectorState) ConnectorState
        +health() ConnectorHealth
    }
    class BaseParser {
        <<abstract>>
        +name: str
        +version: str
        +supports(mime: str, ext: str) bool
        +parse(obj: FetchedObject) UnifiedDocument
    }
    class UnifiedDocument
    class FetchedObject
    class SourceItem
    BaseConnector <|-- PDFConnector
    BaseConnector <|-- MarkdownConnector
    BaseConnector <|-- HTMLConnector
    BaseConnector <|-- GitHubConnector
    BaseParser <|-- PDFParser
    BaseParser <|-- MarkdownParser
    BaseParser <|-- HTMLParser
    BaseParser <|-- GitHubDocParser
    BaseConnector --> SourceItem
    BaseConnector --> FetchedObject
    BaseParser --> UnifiedDocument
```

### 3.2.1 Connector Interface Contract

| Method | Input | Output | Contract |
|---|---|---|---|
| `discover(state)` | Persisted connector state (cursor, ETags, last commit) | Lazy iterator of `SourceItem(uri, canonical_uri, expected_size, source_hint_hash, source_metadata)` | MUST be incremental: yields only items changed since `state`. MUST NOT download bodies. |
| `fetch(item)` | `SourceItem` | `FetchedObject(bytes, mime_type, sha256, size, fetched_at, source_metadata)` | MUST compute sha256. MUST honour timeouts and retries (`tenacity`, exponential backoff 1s→32s, 5 attempts). |
| `checkpoint(state)` | State after successful Bronze commit | New state | State persisted in Postgres `connector_state` keyed by `(connector_name, config_id)`. MUST only advance after Bronze write succeeds. |
| `health()` | — | `ConnectorHealth(ok, latency_ms, detail)` | Used by `/health/connectors`. |

### 3.2.2 Unified Document Object

`akl.ingestion.models.UnifiedDocument` (Pydantic, frozen):

| Field | Type | Notes |
|---|---|---|
| `document_id` | UUID | |
| `content_sha256` | str | |
| `source_type` | Literal | |
| `canonical_source_uri` | str | |
| `title` | str | |
| `language` | str | |
| `text` | str | Canonical Markdown-flavoured text |
| `blocks` | list[Block] | Ordered structural blocks: `HeadingBlock(level,text)`, `ParagraphBlock`, `CodeBlock(language)`, `TableBlock(markdown, rows, cols)`, `ListBlock(items, ordered)`, `ImageBlock(alt, meta)`, `PageBreakBlock(page)` — each with `start_char`, `end_char` |
| `structure` | list[HeadingNode] | Derived tree |
| `tables`, `code_blocks`, `images`, `page_map` | lists | Derived indexes into `blocks` |
| `quality` | QualityReport | score + flags |
| `fingerprint_simhash` | int | |
| `security_level`, `allowed_groups` | | From connector config / path rules |
| `metadata` | dict[str,str] | |
| `parser_name`, `parser_version` | str | |

The `blocks` list is the single input to the Chunking Engine (Chapter 4). All parsers MUST produce it; chunking never re-parses text.

## 3.3 Connectors

### 3.3.1 PDF Connector

- Modes: `upload` (API writes Bronze directly and enqueues) and `directory` (watches `AKL_PDF_INBOX_PATH` mounted volume / S3 prefix).
- `discover`: lists inbox; state = set of processed `(path, mtime, size)`; yields items not in state.
- `fetch`: reads bytes; validates `%PDF-` magic; computes sha.
- Config: `inbox_path`, `security_level_default`, `path_rules: [{glob, security_level, allowed_groups}]`.

### 3.3.2 Markdown Connector

- Sources: local directory or S3 prefix of `.md`/`.mdx`.
- Frontmatter (YAML between `---`) parsed for `title`, `tags`, `security_level`, `owners`.
- Same state model as PDF connector.

### 3.3.3 HTML Connector

- Sources: sitemap URL, URL list file, or crawl seed with `allow_patterns`, `max_depth (default 2)`, `max_pages`, `same_host_only=true`.
- `discover`: fetches sitemap / crawls link graph using conditional GET (`If-None-Match`, `If-Modified-Since`) with state of ETag/Last-Modified per URL; yields only changed URLs.
- `robots.txt` MUST be honoured; `AKL_HTML_USER_AGENT` identifies the crawler.
- Rate limit: `requests_per_second` per host (default 2).

### 3.3.4 GitHub Connector

- Config: `owner`, `repo`, `branch`, `include_globs (default ["**/*.md","**/*.mdx","**/*.rst","**/*.txt","docs/**"])`, `exclude_globs (["**/node_modules/**","**/vendor/**","**/CHANGELOG*"])`, `include_code: false`, `code_globs`, `max_file_bytes (1 MiB)`, `security_level`, `allowed_groups`.
- Strategy selection:
  - **API mode** (default): `GET /repos/{o}/{r}/branches/{b}` → head sha; if equals `state.last_commit_sha` → zero items. Else `GET /repos/{o}/{r}/git/trees/{sha}?recursive=1` (single call, up to 100k entries) → write `bronze/github_snapshots`; diff against previous snapshot on `(path, blob_sha)` → added/modified/deleted; fetch blobs via `GET /repos/{o}/{r}/git/blobs/{blob_sha}` (base64) only for added/modified. Deleted paths emit `DeletionEvent` → tombstone flow (Chapter 9.15).
  - **Clone mode** (fallback when tree > 100k entries or rate limit exhausted): `git clone --depth 1 --filter=blob:none --branch <b>` into task temp dir, `git ls-tree -r`, then sparse checkout of matching paths.
- Canonical URI: `github://owner/repo/<branch>/<path>` (commit-agnostic); `source_uri` retains commit sha for exact provenance.
- Rate-limit handling: read `X-RateLimit-Remaining`; if `< 50`, sleep until `X-RateLimit-Reset`; emit `akl_github_rate_limit_remaining` gauge.

## 3.4 Parsing

### 3.4.1 Parser Selection

`ParserRegistry.select(mime_type, extension, source_type)` — ordered rules; first match wins; no match → quarantine `AKL-E3003 UNSUPPORTED_FORMAT`.

### 3.4.2 PDF Parsing Strategy

Library: `pymupdf` (fitz) primary; `pdfplumber` for table extraction on pages flagged as table-dense.

Algorithm per page:

1. Extract text blocks with bounding boxes and font sizes (`page.get_text("dict")`).
2. **Heading detection**: compute font-size histogram for document; body size = mode; block is heading if `size ≥ body × 1.15` and `len(text) < 120` and not ending with `.`; heading level derived by rank of size (max 4 levels). Bold-only detection as secondary signal.
3. **Reading order**: sort blocks by column (x-cluster via 1-D k-means with k∈{1,2}) then y.
4. **Header/footer removal**: text blocks appearing at same y-band (±5 pt) with ≥ 80% textual similarity on ≥ 60% of pages are boilerplate → removed, recorded in `quality_flags`.
5. **Hyphenation repair**: join `word-\nword` when `word` continuation exists in dictionary or is lowercase continuation.
6. **Table extraction**: if page has ≥ 3 horizontal rules or ≥ 8 aligned x-positions, run `pdfplumber.extract_tables()`; convert to Markdown table; replace region text with `TableBlock`.
7. **Code detection**: monospace font family (Courier, Mono, Consolas) blocks → `CodeBlock(language=None)`; language guessed with `guesslang`-style heuristics (keywords) or left null.
8. **Image metadata**: `page.get_images()` → `ImageBlock(alt=None, width, height, page)`; if page has 0 text chars and ≥ 1 image → `image_only_page` flag. OCR is a non-goal.
9. Emit `PageBreakBlock(page)` between pages; build `page_map`.

Failure scenarios: encrypted PDF (`AKL-E3010`, quarantine unless `AKL_PDF_TRY_EMPTY_PASSWORD=true` succeeds), corrupted xref (`AKL-E3011`, try `pymupdf` repair, else quarantine), > `AKL_PDF_MAX_PAGES (2000)` (`AKL-E3012`, quarantine), zero extractable text (`AKL-E3013`, quarantine with `image_only` reason).

### 3.4.3 Markdown AST Parsing

Library: `markdown-it-py` with plugins `front_matter`, `table`, `footnote`, `tasklists`; produces a token stream converted to an AST.

- Headings → `HeadingBlock(level)`; ATX and setext both supported.
- Fenced/indented code → `CodeBlock(language=info string)`.
- Tables → `TableBlock` with normalised pipe syntax.
- Lists → `ListBlock` (nested lists flattened to depth-annotated items).
- Inline HTML → passed through HTML cleaner (3.4.4) inline.
- Links preserved as `[text](url)`; images → `ImageBlock(alt)`.
- MDX: JSX tags stripped; `import/export` lines removed.
- Text canonicalisation: CRLF→LF, tabs→4 spaces in code only, collapse >2 blank lines, NFC Unicode normalisation.

### 3.4.4 HTML Cleaning

Library: `selectolax` (fast) for DOM; `trafilatura` for main-content extraction as a first pass; fallback to heuristic density-based extraction.

1. Remove `script, style, noscript, iframe, svg, nav, footer, header, aside, form, [role=navigation], [aria-hidden=true], .cookie*, .sidebar, .breadcrumb*`.
2. Main content: `trafilatura.extract(include_tables=True, include_links=True, output_format="markdown")`; if result `< 200 chars` and page text `> 1000 chars`, fall back to the largest text-density subtree.
3. Convert remaining DOM to Markdown via `markdownify` with heading/table/code mapping; `<pre><code class="language-x">` → `CodeBlock(x)`.
4. Canonical URL from `<link rel=canonical>` if same host.
5. Title: `<title>` → `og:title` → first `<h1>`.
6. Boilerplate detection across pages of one host: shingle-based repeated-block detection (blocks appearing on ≥ 30% of pages) removed.

### 3.4.5 GitHub Repository Documents

GitHub blobs are dispatched by extension to Markdown (`.md/.mdx`), reStructuredText (`docutils` → HTML → HTML parser), plain text, or (when `include_code=true`) the **Code parser**: `tree-sitter` (Python, TypeScript, Go, Java, Rust grammars) extracts top-level functions/classes with docstrings; each definition becomes a `CodeBlock` with `metadata.symbol`. Repo-level metadata (`repo`, `branch`, `path`, `commit_sha`, `last_author`, `last_commit_at`) is attached.

### 3.4.6 Language Detection

`lingua-language-detector` on the first 5,000 chars of prose blocks (code excluded). Confidence < 0.7 → `language="und"`. Stored on document and chunks.

### 3.4.7 Metadata Extraction

Common: title, language, word/char counts, heading count, table count, code block count, image count, created/modified timestamps from source, author (PDF metadata `author`, frontmatter `owners`, git last author). All under `metadata` map with namespaced keys (`pdf.author`, `git.last_author`, `html.og_site_name`).

### 3.4.8 Checksums, Versions, Fingerprints

| Artefact | Algorithm | Purpose |
|---|---|---|
| `content_sha256` | SHA-256 of raw bytes | Bronze identity, exact dedup |
| `text_sha256` | SHA-256 of canonical text | Detect parser-output change without byte change (parser upgrade) |
| `document_version_id` | uuid5 over `(document_id, content_sha256, parser_version)` | Silver row identity |
| `fingerprint_simhash` | 64-bit SimHash over 5-gram word shingles of canonical text | Near-dup detection |
| `chunk_checksum` | SHA-256 of NFC-normalised, whitespace-collapsed chunk text | Embedding cache key |

## 3.5 Validation Rules

Executed by `akl.ingestion.validators` before Silver write; each rule has a code and severity (`reject` → quarantine, `warn` → flag).

| Rule | Severity | Code |
|---|---|---|
| Size within `[AKL_MIN_DOC_BYTES=64, AKL_MAX_DOC_BYTES=50 MiB]` | reject | AKL-E3001 |
| MIME/extension supported | reject | AKL-E3003 |
| Not a binary masquerading as text (null-byte ratio > 0.01) | reject | AKL-E3004 |
| Canonical text ≥ 100 chars | reject | AKL-E3005 |
| Text density (alnum chars / total) ≥ 0.4 | warn | AKL-W3006 |
| Language detected with confidence ≥ 0.7 | warn | AKL-W3007 |
| No secrets detected (regex set: AWS keys, private key headers, JWT-like tokens, generic `api_key=` patterns) | reject unless `AKL_ALLOW_SECRET_LIKE_CONTENT=true` | AKL-E3008 |
| Security level assigned | reject | AKL-E3009 |
| Heading tree depth ≤ 8 | warn | AKL-W3014 |

## 3.6 Quarantine Flow

```mermaid
sequenceDiagram
    participant V as Validator/Parser
    participant Q as Quarantine Writer
    participant M as MinIO
    participant PG as Postgres
    participant PR as Prometheus

    V->>Q: quarantine(fetched_object, error_code, detail, stage)
    Q->>M: PUT quarantine/ingest_date=.../objects/sha256=<h>.<ext>
    Q->>M: append quarantine/ingest_date=.../reasons/part-<run>.parquet
    Q->>PG: INSERT quarantine_items (document_id, sha, code, stage, detail, run_id, status='open')
    Q->>PR: akl_quarantine_total{code,stage,source_type}++
    Q-->>V: QuarantineReceipt
```

Admin can `GET /v1/admin/quarantine`, `POST /v1/admin/quarantine/{id}/retry` (re-runs parse with current parser version) or `.../dismiss`. Items older than `AKL_QUARANTINE_RETENTION_DAYS (90)` are purged by the Maintenance DAG.

## 3.7 Bronze Write Flow

```mermaid
sequenceDiagram
    participant C as Connector
    participant BW as Bronze Writer
    participant M as MinIO
    participant PG as Postgres

    C->>BW: commit(fetched_object, source_item, run_id)
    BW->>M: HEAD bronze/raw/source_type=t/sha256=h.ext
    alt exists
        BW->>BW: skip body upload (dedup); metric akl_bronze_dedup_hits_total++
    else
        BW->>M: PUT object (Content-MD5, x-amz-meta-sha256)
    end
    BW->>BW: buffer manifest row
    BW->>PG: UPSERT documents (document_id, canonical_uri, latest_sha, status='bronze')
    BW->>PG: INSERT document_versions (document_id, content_sha256, bronze_key, run_id) ON CONFLICT DO NOTHING
    Note over BW,M: at task end
    BW->>M: PUT bronze/manifest/ingest_date=d/part-<run_id>-<n>.parquet
    BW->>C: checkpoint(state)
```

Ordering guarantee: raw object → Postgres version row → manifest file → connector checkpoint. A crash between steps yields at worst an orphan raw object (harmless, content-addressed) or a manifest missing rows that the next run re-discovers because the checkpoint did not advance.

## 3.8 Ingestion Sequence (End-to-End)

```mermaid
sequenceDiagram
    participant AF as Airflow akl_ingestion
    participant CN as Connector
    participant BW as Bronze Writer
    participant PS as Parser
    participant VL as Validator
    participant SW as Silver Writer
    participant QW as Quarantine

    AF->>CN: discover(state)
    loop each SourceItem (parallel, bounded)
        CN->>CN: fetch(item)
        CN->>BW: commit
    end
    AF->>AF: list new (document_id, sha) from Postgres where status='bronze'
    loop each new version
        AF->>PS: parse(bronze bytes)
        alt parse error
            PS->>QW: quarantine(AKL-E30xx)
        else
            PS->>VL: validate(UnifiedDocument)
            alt reject
                VL->>QW: quarantine
            else
                VL->>SW: write silver.documents row; update Postgres status='silver'
            end
        end
    end
    AF->>AF: dedup pass (SimHash) over new documents vs ledger
    AF->>AF: emit metrics, trigger akl_chunking
```

## 3.9 Failure Scenarios Matrix

| Scenario | Detection | Handling | Code |
|---|---|---|---|
| Source unreachable | HTTP/SSH error after retries | Task fails; Airflow retries; state not advanced | AKL-E3020 |
| Partial fetch (size mismatch) | `len(bytes) != Content-Length` | Discard, retry | AKL-E3021 |
| Bronze write fails | S3 exception | Task fails; no Postgres row written | AKL-E3022 |
| Postgres unavailable | connection error | Task fails; raw object may exist (orphan OK) | AKL-E3023 |
| Parser exception | any | Quarantine with traceback hash | AKL-E3030 |
| Parser timeout (> `AKL_PARSE_TIMEOUT_S=120`) | `signal.alarm`/subprocess | Quarantine | AKL-E3031 |
| Memory blow-up (huge PDF) | RSS > `AKL_PARSE_MAX_RSS_MB` | Parse in subprocess with `resource` limits; quarantine on kill | AKL-E3032 |
| Duplicate content, different URI | sha exists | Manifest row added; no Silver reparse; `documents` row for new URI points to same version | — |

## 3.10 Ingestion Folder Structure

```
akl/ingestion/
├── __init__.py
├── registry.py            # ConnectorRegistry, ParserRegistry
├── models.py              # SourceItem, FetchedObject, UnifiedDocument, Block types, QualityReport
├── state.py               # ConnectorState persistence (Postgres connector_state)
├── bronze_writer.py       # content-addressed writes, manifest buffering
├── silver_writer.py       # documents.parquet writes, Postgres status transitions
├── quarantine.py          # QuarantineWriter, retry/dismiss
├── validators.py          # rule engine
├── dedup.py               # SimHash, ledger
├── fingerprint.py         # sha256/simhash helpers
├── language.py            # detection wrapper
├── connectors/
│   ├── base.py
│   ├── pdf.py
│   ├── markdown.py
│   ├── html.py
│   └── github.py
└── parsers/
    ├── base.py
    ├── pdf.py
    ├── markdown.py
    ├── html.py
    ├── rst.py
    ├── code.py            # tree-sitter based
    └── boilerplate.py     # cross-document repeated block removal
```

## 3.11 Document Quality Score

`quality_score = clamp(0,1, 0.35·text_density + 0.20·structure_score + 0.15·language_conf + 0.15·(1 − boilerplate_ratio) + 0.15·length_score)` where `structure_score = min(1, headings/ (words/400))`, `length_score = min(1, words/300)`. Documents `< AKL_DOC_QUALITY_MIN (0.35)` are kept in Silver but excluded from Gold with flag `low_quality`.

---

# Chapter 4 — Chunking Engine

## 4.1 Purpose and Constraints

The chunking engine transforms the ordered `blocks` of a `UnifiedDocument` into retrieval units that (a) fit the embedding model's effective context (`bge-small-en-v1.5` max 512 tokens; quality degrades past ~350), (b) preserve semantic coherence, (c) never split atomic structures (code blocks, table rows) unless unavoidable, (d) carry enough metadata to cite precisely, and (e) are stable across small document edits.

| Parameter | Default | Env |
|---|---|---|
| Target tokens per chunk | 320 | `AKL_CHUNK_TARGET_TOKENS` |
| Max tokens per chunk (hard) | 448 | `AKL_CHUNK_MAX_TOKENS` |
| Min tokens per chunk | 64 | `AKL_CHUNK_MIN_TOKENS` |
| Overlap tokens (prose) | 48 | `AKL_CHUNK_OVERLAP_TOKENS` |
| Semantic split threshold (cosine drop) | 0.25 | `AKL_CHUNK_SEMANTIC_THRESHOLD` |
| Code chunk max tokens | 400 | `AKL_CHUNK_CODE_MAX_TOKENS` |
| Table max tokens before row-splitting | 400 | `AKL_CHUNK_TABLE_MAX_TOKENS` |
| Context prefix max tokens | 40 | `AKL_CHUNK_CONTEXT_PREFIX_TOKENS` |

All parameters are loaded from `configs/chunking.yaml`; `chunk_config_hash = sha256(canonical_yaml)[:16]` is stored per chunk so a config change is detected as a re-chunk trigger.

## 4.2 Hybrid Hierarchical Chunking

Three passes, applied in order:

1. **Structural pass (heading-aware)** — Partition `blocks` into *sections* at `HeadingBlock`s. Each section carries `heading_path` (stack of ancestor headings). Sections are the parent units (`parent_chunk_id` refers to a synthetic section chunk of type `heading_only` when `AKL_CHUNK_EMIT_SECTION_PARENTS=true`, used for hierarchical retrieval in Enterprise Scale).
2. **Semantic pass** — Within a section, group consecutive prose blocks into candidate chunks by greedily accumulating sentences until `target_tokens`, but *cut early* at sentence boundaries where the cosine similarity between the running chunk embedding (cheap: mean of sentence embeddings from bge-small, computed in batch) and the next sentence drops below `1 − AKL_CHUNK_SEMANTIC_THRESHOLD`. Semantic pass is optional (`AKL_CHUNK_SEMANTIC_ENABLED`, default true) because it costs one embedding pass per document; when disabled, pass 3 alone applies.
3. **Token pass (hard limits)** — Any candidate exceeding `max_tokens` is split at sentence → clause → whitespace boundaries with overlap. Any candidate below `min_tokens` is merged with its neighbour in the same section (Chapter 4.11).

```mermaid
flowchart TD
    A[UnifiedDocument.blocks] --> B[Structural pass: sections by heading]
    B --> C{block type}
    C -->|prose/list| D[Semantic grouping by sentence]
    C -->|code| E[Code-aware splitter]
    C -->|table| F[Table-aware splitter]
    D --> G[Token pass: split > max, merge < min, apply overlap]
    E --> G
    F --> G
    G --> H[Attach heading_path, context_prefix, offsets, pages]
    H --> I[Compute chunk_key, chunk_checksum, chunk_id, quality]
    I --> J[Link prev/next/parent]
    J --> K[silver/chunks rows]
```

## 4.3 Heading-Aware Rules

- A chunk never spans two sibling sections.
- A section whose total tokens ≤ `max_tokens` becomes exactly one chunk (no fragmentation of short sections).
- A heading with no body text is merged into the next section's `heading_path` (no `heading_only` chunk emitted unless section parents are enabled).
- `heading_path` is truncated to the last 4 levels for `context_prefix` rendering; full path retained in metadata.

## 4.4 Token-Aware Rules

Token counting uses the exact `bge-small-en-v1.5` tokenizer (`tokenizers` Rust library, `BertWordPieceTokenizer`), cached per process. Counting is applied to `context_prefix + "\n" + text` because that is what is embedded; the stored `token_count` is for `text` alone and `embedded_token_count` is stored separately in Gold.

## 4.5 Code-Aware Chunking

- A `CodeBlock` ≤ `code_max_tokens` is one chunk of type `code`; the preceding paragraph (if ≤ 80 tokens) is attached as `lead_in` inside the chunk text to preserve the explanation-code pairing.
- Larger code blocks are split at top-level definition boundaries using `tree-sitter` when language is known; else at blank-line groups; else at line count. Each split chunk repeats the first line (signature/comment) as context and records `code_split_index/total`.
- Code chunks have zero overlap (duplicated code harms BM25 precision).
- `code_language` propagates from the fence info string or parser detection.

## 4.6 Table-Aware Chunking

- A table ≤ `table_max_tokens` is one chunk of type `table` (Markdown form), with the caption / preceding sentence attached.
- Larger tables are split by rows; **every split repeats the header row** and records `table_split_index/total`. Column count is preserved; cells are never truncated.
- Tables with ≥ 30 columns are transposed into row-wise key:value text before chunking (wide tables embed poorly).

## 4.7 Overlap Strategy

- Prose: sliding overlap of `overlap_tokens` **sentence-aligned** (the overlap is the trailing sentences of the previous chunk that together are ≤ overlap_tokens; never mid-sentence).
- Overlap is applied *within a section only*.
- `overlap_prev_tokens` stored per chunk so the context builder can trim duplicated text when adjacent chunks are both retrieved (Chapter 6.10).
- Code and table chunks: no overlap.

## 4.8 Chunk Metadata

Every chunk row carries the fields in Chapter 2.5.2. Additionally, the `context_prefix` is rendered as:

```
<title> › <heading 1> › <heading 2> › <heading 3>
```

and prepended at embedding time only (not part of `text`, not part of `chunk_checksum`). Rationale: the checksum must be stable when a document is retitled but the chunk body is unchanged; the embedding cache key therefore uses `embedded_text_sha256` (which includes the prefix) in Gold, while `chunk_checksum` drives lineage. The cache lookup key is `embedded_text_sha256`.

## 4.9 Chunk Quality Score

```
quality = clamp(0, 1,
    0.30 · length_fit          # 1 at target_tokens, linearly down to 0 at min or 1.5×max
  + 0.20 · alnum_ratio         # alphanumeric chars / total
  + 0.15 · sentence_completeness  # 1 if starts uppercase/ends terminal punct (prose); 1 for code/table
  + 0.15 · (1 − boilerplate_ratio)  # fraction of tokens matching repeated-block ledger
  + 0.10 · heading_context     # 1 if heading_path non-empty
  + 0.10 · (1 − repetition)    # 1 − (duplicate 3-gram ratio)
)
```

Thresholds: `quality < AKL_CHUNK_QUALITY_MIN (0.30)` → flag `low_quality`, excluded from Gold. `0.30 ≤ q < 0.50` → included, flagged `marginal`, down-weighted ×0.9 in fusion.

## 4.10 Chunk Identity and Versioning

Per ADR-003:

```
ordinal            = index of chunk within its section (0-based)
chunk_key          = sha1(document_id || "\x1f" || "/".join(heading_path) || "\x1f" || ordinal)[:40]
normalized_text    = NFC(text) with runs of whitespace collapsed to single space, trimmed
chunk_checksum     = sha256(normalized_text)
chunk_id           = uuid5(AKL_NS_CHUNK, f"{document_id}:{chunk_key}:{chunk_checksum}")
```

Versioning semantics:

| Event | chunk_key | chunk_checksum | chunk_id | Action |
|---|---|---|---|---|
| Unchanged | same | same | same | Skip |
| Text edited in place | same | differs | new | New chunk row; old row `is_current=false`; embedding for new checksum; Qdrant upsert new id, delete old id |
| Heading renamed | differs | same | new | New row; embedding **cache hit** on text (if prefix unchanged) → cheap; Qdrant upsert/delete |
| Chunk inserted mid-section | later ordinals shift → keys differ | same for shifted | new ids | Shifted chunks re-keyed but embeddings hit cache (same `embedded_text_sha256` when prefix unchanged) |
| Chunk removed | key vanishes | — | — | Old row tombstoned; Qdrant delete |

Ordinal shifting is the known weakness of heading+ordinal keys. Mitigation (Chapter 4.12 step 4): after computing new chunks, the algorithm attempts **checksum-based re-association** so a shifted-but-identical chunk inherits the *lineage* (`lineage_id`, a separate column defaulting to the first `chunk_id` ever observed for that content within the document) even though its `chunk_key` changed. `lineage_id` is what citations link to in conversation history.

## 4.11 Merge and Split Algorithms

**Split** (`split_oversized(candidate, max_tokens, overlap)`):

```
sentences = sentence_split(candidate.text)          # pysbd, language-aware
if len(sentences) == 1: sentences = clause_split(candidate.text)   # on ; : , — with ≥ 12 tokens per clause
if still single unit: sentences = whitespace_split(candidate.text, window=max_tokens - overlap)
chunks = []; buf = []; buf_tokens = 0
for s in sentences:
    t = tokens(s)
    if buf_tokens + t > max_tokens and buf:
        chunks.append(join(buf))
        buf = tail_sentences(buf, overlap_tokens)   # sentence-aligned overlap
        buf_tokens = tokens(buf)
    buf.append(s); buf_tokens += t
if buf: chunks.append(join(buf))
```

**Merge** (`merge_undersized(chunks_in_section, min_tokens, max_tokens)`):

```
i = 0
while i < len(chunks):
    c = chunks[i]
    if c.tokens < min_tokens and c.type == prose:
        # prefer merging forward with a prose neighbour; else backward
        if i+1 < len and chunks[i+1].type == prose and c.tokens + chunks[i+1].tokens <= max_tokens:
            chunks[i+1] = concat(c, chunks[i+1]); del chunks[i]; continue
        elif i > 0 and chunks[i-1].type == prose and c.tokens + chunks[i-1].tokens <= max_tokens:
            chunks[i-1] = concat(chunks[i-1], c); del chunks[i]; continue
        else:
            c.flags.add("short")      # kept; quality penalised
    i += 1
```

Code/table chunks are never merged with prose except the `lead_in` attachment defined in 4.5/4.6.

## 4.12 Incremental Chunk Update Algorithm

Triggered per `document_version_id` that is current and has no chunks for `(chunker_version, chunk_config_hash)`.

```
new_chunks   = chunk(document)                                     # full re-chunk of this document only
old_chunks   = load_current_chunks(document_id)                    # from Postgres/Silver
old_by_key   = {c.chunk_key: c for c in old_chunks}
old_by_sum   = index old_chunks by chunk_checksum (list)
for n in new_chunks:
    if n.chunk_key in old_by_key and old_by_key[n.chunk_key].chunk_checksum == n.chunk_checksum:
        n.status = UNCHANGED; n.lineage_id = old.lineage_id; reuse chunk_id
    elif n.chunk_key in old_by_key:
        n.status = MODIFIED;  n.lineage_id = old.lineage_id
    elif n.chunk_checksum in old_by_sum:                            # shifted/re-headed identical text
        n.status = MOVED;     n.lineage_id = old_by_sum[..].lineage_id
    else:
        n.status = ADDED;     n.lineage_id = n.chunk_id
removed = old_chunks whose chunk_id not in {n.chunk_id for n in new_chunks}
write silver.chunks rows for MODIFIED|MOVED|ADDED with is_current=true
mark removed + superseded rows is_current=false (Postgres) and append tombstone rows (Parquet)
record ChunkDiff(document_id, unchanged, modified, moved, added, removed) → metrics + XCom
```

Chunking is deterministic; therefore for an UNCHANGED classification the recomputed `chunk_id` equals the stored one and no write occurs. A no-change re-run therefore writes zero rows (NFR-08).

## 4.13 Context Preservation

- `context_prefix` (breadcrumb) is embedded with the chunk (proven to improve retrieval on section-scoped queries).
- `lead_in` sentence attached to code/table chunks.
- `prev_chunk_id`/`next_chunk_id` allow the context builder to fetch neighbours (`AKL_RAG_NEIGHBOR_EXPANSION`, default 0; Enterprise Scale: 1).
- `parent_chunk_id` allows "small-to-big" retrieval (retrieve on chunk, present section) as a future toggle.

## 4.14 Citation Mapping

Each chunk stores `start_char/end_char` into `silver.documents.text` and `page_start/page_end` for PDFs, plus `repo/branch/path` and, when available, `line_start/line_end` (computed from character offsets for Markdown and code sources). A citation renders as:

| Source | Locator format |
|---|---|
| PDF | `<title>, p. <page_start>[–<page_end>]` |
| Markdown / GitHub | `<repo>/<path>#L<line_start>-L<line_end> @ <branch>` (with commit sha in trace) |
| HTML | `<canonical_url>` + `#:~:text=<first 8 words>` text fragment |

## 4.15 Worked Example

Input section (Markdown, heading path `["Deployment","Docker Compose"]`, 2 paragraphs of 180 and 210 tokens, then a 120-token YAML block, then a 4-row table):

- Paragraph 1 (180) + Paragraph 2 (210) = 390 > target 320 → semantic pass finds a topic shift between them (cosine drop 0.31 > 0.25) → two prose chunks: `[P1]`, `[P2 with 48-token overlap tail of P1]`.
- YAML block (120) → one `code` chunk with `lead_in` = last sentence of P2 (if ≤ 80 tokens), `code_language=yaml`.
- Table (4 rows, 90 tokens) → one `table` chunk with caption.
- Four chunks emitted, ordinals 0–3, `heading_path` identical, keys `sha1(doc|Deployment/Docker Compose|0..3)`.

## 4.16 Chunking Folder Structure

```
akl/chunking/
├── __init__.py
├── config.py            # ChunkConfig (pydantic) loaded from configs/chunking.yaml, config hash
├── tokenizer.py         # cached bge tokenizer, count(), tail_sentences()
├── sentences.py         # pysbd wrapper, clause splitter
├── structural.py        # sections from blocks, heading_path stack
├── semantic.py          # sentence embeddings + boundary detection
├── code_splitter.py     # tree-sitter aware
├── table_splitter.py    # header-repeating row splitter, transpose
├── merge_split.py       # algorithms from 4.11
├── identity.py          # chunk_key, checksum, chunk_id, lineage
├── quality.py           # chunk quality score
├── incremental.py       # diff algorithm 4.12
├── engine.py            # ChunkingEngine.chunk(document) -> list[Chunk]
└── models.py            # Chunk, ChunkDiff
```

---

# Chapter 5 — Embedding Pipeline

## 5.1 Embedding Architecture

```mermaid
flowchart LR
    A[v_embedding_coverage<br/>chunks lacking vector for current embedding_version] --> B[Batcher<br/>sort by embedded_token_count, batch 64]
    B --> C{Cache lookup<br/>embedded_text_sha256 + model}
    C -->|hit| D[Reuse vector]
    C -->|miss| E[Embedding Service<br/>ONNX bge-small, CPU threads=N]
    E --> F[L2 normalize]
    F --> G[Write cache rows]
    D --> H[gold/chunk_embeddings append]
    G --> H
    H --> I[MLflow run: counts, throughput, cache hit rate, norm stats]
    H --> J[Trigger akl_qdrant_sync]
```

## 5.2 Embedding Lifecycle

| Stage | Trigger | Output |
|---|---|---|
| Backlog detection | `akl_embedding` DAG start | List of `(chunk_id, embedded_text_sha256)` with no vector for `AKL_EMBEDDING_VERSION` |
| Cache resolution | per batch | Split into hits/misses |
| Generation | misses | float32[384] normalised |
| Persistence | per batch | Gold Parquet + Postgres `embedding_cache` |
| Sync | DAG trigger | Qdrant upserts (Chapter 5.14) |
| Retirement | model change | Old `embedding_version` partition retained `AKL_EMBEDDING_RETIRE_DAYS (30)` then deleted |

## 5.3 Why BGE-small-en-v1.5

| Criterion | bge-small-en-v1.5 | Rationale |
|---|---|---|
| Dimension | 384 | 4× smaller than 1536-d models → Qdrant memory and Parquet size 4× lower |
| Parameters | 33 M | CPU inference ≈ 200–400 chunks/s with ONNX on 4 cores |
| MTEB retrieval | ≈ 51.7 nDCG@10 | Competitive with much larger models on English retrieval |
| Context | 512 tokens | Matches chunk max 448 + prefix |
| Licence | MIT | Free for enterprise use |
| Query instruction | `Represent this sentence for searching relevant passages: ` | Asymmetric; applied to queries only |

### 5.3.1 Alternatives Comparison

| Model | Dim | Params | CPU speed | Quality | Verdict |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 22 M | fastest | lower retrieval quality | Rejected |
| `bge-base-en-v1.5` | 768 | 109 M | 3× slower | +2 nDCG | Enterprise Scale candidate with GPU |
| `e5-small-v2` | 384 | 33 M | similar | similar | Viable; BGE chosen for instruction-tuned queries |
| `nomic-embed-text-v1.5` | 768 (Matryoshka) | 137 M | slower | higher, 8k ctx | Future: long-context chunks |
| OpenAI `text-embedding-3-small` | 1536 | — | API | high | Rejected: cost, egress, non-local |

The abstraction `akl.embedding.provider.EmbeddingProvider` (methods `embed_documents(texts)`, `embed_query(text)`, `model_id`, `model_version`, `dim`) permits swapping; every vector is tagged with `embedding_version`.

## 5.4 Embedding Cache

**Storage**: Postgres table `embedding_cache(embedded_text_sha256, model_id, model_version, dim, vector BYTEA (float32 LE), created_at, hit_count)` with PK on the first three columns. Vector stored as 1,536-byte BYTEA (384 × 4). At 100k chunks ≈ 150 MB — acceptable in Postgres for MVP.

**Enterprise Scale**: cache index in Postgres (hash → Gold file/row pointer) and payload in Gold Parquet only; or a KV store (Redis/RocksDB) — Chapter 14.

**Eviction**: rows not hit in `AKL_EMBEDDING_CACHE_TTL_DAYS (180)` are deleted by the Maintenance DAG. Cache hit metric: `akl_embedding_cache_hits_total / (hits + misses)`.

## 5.5 Embedding Versioning

`embedding_version = f"{model_id_slug}__{model_version}__{dim}"` e.g. `bge-small-en-v1.5__1.5__384`. Also included: `embedder_version` (code semver) and `normalize=true`. Qdrant collection name embeds the version family: `kb_chunks_v1` corresponds to the 384-d cosine family; a dimension change requires `kb_chunks_v2` and alias cutover.

## 5.6 Batch Embedding Generation

- Sort backlog by token length (reduces padding waste up to 40%).
- Batch size `AKL_EMBED_BATCH_SIZE` (default 64 CPU / 256 GPU).
- ONNX Runtime with `intra_op_num_threads = AKL_EMBED_THREADS (default cores−1)`; INT8 dynamic quantisation optional (`AKL_EMBED_ONNX_INT8=true`) giving ~1.8× speed at < 0.5% nDCG loss.
- Max sequence length 512; truncation is an error at this stage (chunker guarantees fit) → `AKL-E5003`.
- Airflow parallelism: backlog split into `AKL_EMBED_TASK_SHARDS (4)` dynamic-mapped tasks by `hash(chunk_id) % shards`.

### 5.6.1 GPU vs CPU

| Aspect | CPU (MVP) | GPU (Enterprise) |
|---|---|---|
| Runtime | ONNX Runtime CPU EP | ONNX CUDA EP / PyTorch fp16 |
| Batch | 64 | 256–1024 |
| Throughput | 200–400 chunks/s | 5k–20k chunks/s (T4→A10) |
| Detection | `AKL_EMBED_DEVICE=auto` → `cuda` if available | |
| Cost lever | INT8, thread pinning | Mixed precision, multi-GPU sharding |

## 5.7 Embedding Metadata Schema

See Gold `chunk_embeddings` (Chapter 2.6.2) and Postgres `embedding_jobs(job_id, run_id, embedding_version, shard, chunks_total, cache_hits, generated, failed, started_at, finished_at, throughput_cps, mlflow_run_id)`.

## 5.8 Qdrant Collection Schema

| Setting | Value | Rationale |
|---|---|---|
| Collection | `kb_chunks_v1` (alias `kb_chunks`) | Alias enables blue/green reindex |
| Vector size | 384 | |
| Distance | `Cosine` | Vectors normalised → equivalent to dot; cosine chosen for readability of scores |
| HNSW `m` | 16 | Default balance |
| HNSW `ef_construct` | 128 | Higher build quality; build cost acceptable |
| `hnsw_ef` (search) | 128 (configurable per request) | Recall/latency |
| `on_disk_payload` | true | Payload includes `text`; keep RAM for vectors |
| Quantisation | none (MVP); scalar int8 (Enterprise) | |
| Shards | 1 (MVP); N (Enterprise) | |
| Replication | 1; 2+ | |
| Payload indexes | `document_id`(keyword), `source_type`(keyword), `security_level`(keyword), `allowed_groups`(keyword), `chunk_type`(keyword), `code_language`(keyword), `repo`(keyword), `token_count`(integer), `document_updated_at`(integer), `quality_score`(float) | Filterable fields |
| Point ID | `chunk_id` UUID | Stable, matches Gold |

## 5.9 Vector Dimension, Similarity Metrics

- 384-d float32 = 1,536 B per vector; 100k vectors = 154 MB raw; HNSW graph overhead ≈ +50% → ~230 MB RAM. 1B vectors ≈ 1.5 TB raw → sharding and quantisation mandatory (Chapter 14).
- **Cosine**: `cos(a,b) = a·b / (‖a‖‖b‖)`; scale-invariant; standard for text embeddings. After L2 normalisation identical to dot product.
- **Dot product**: faster (no norms) but sensitive to magnitude; used internally since vectors are normalised.
- **Euclidean**: `‖a−b‖² = 2 − 2·cos` for unit vectors → same ranking; not used.

## 5.10 HNSW Explained

```mermaid
flowchart TB
    subgraph L2["Layer 2 (sparse, long-range edges)"]
        A2((a)) --- D2((d))
    end
    subgraph L1["Layer 1"]
        A1((a)) --- B1((b)) --- D1((d)) --- F1((f))
    end
    subgraph L0["Layer 0 (all points, short-range edges, m=16)"]
        A0((a)) --- B0((b)) --- C0((c)) --- D0((d)) --- E0((e)) --- F0((f)) --- G0((g))
        B0 --- E0
        C0 --- G0
    end
    Q[query] -.greedy descent.-> A2 -.-> D1 -.-> E0
```

Search enters at the top layer, greedily moves to the nearest neighbour, descends a layer, and repeats; at layer 0 a beam of width `ef` is maintained. Complexity ≈ O(log N). Recall is controlled by `ef`; memory by `m`. Insertions are incremental (no full rebuild), which is why point-level upserts are cheap.

## 5.11 Approximate Nearest Neighbour Search and Metadata Filtering

Qdrant applies filters **during** graph traversal using payload indexes (filterable HNSW), avoiding the recall collapse of post-filtering when the filter is selective (e.g. `repo = X` matching 1% of points). All AKL queries carry at least the security filter; therefore payload indexes on `security_level` and `allowed_groups` are mandatory (`AKL-E5010` if missing at startup health check).

## 5.12 Hybrid Retrieval Scoring, Top-K, Reranking

Covered in Chapter 6.7–6.9; summary: dense top-50 (`hnsw_ef=128`), sparse top-50, RRF fusion `score = Σ 1/(60 + rank_i)`, quality down-weight for `marginal`, rerank top-40 with cross-encoder, present top-8.

## 5.13 Embedding Update, Deletion and Reindex Flows

**Update** (chunk MODIFIED): new `chunk_id` → backlog → embed (or cache) → Gold append → sync upserts new point, deletes old `chunk_id` point.

**Deletion** (document deleted or chunk removed): Silver tombstone → Gold `retrieval_units` row disappears from `v_gold_active_units` → sync computes `qdrant_ids − gold_ids` → batch delete. Embedding rows in Gold are retained (they are keyed by checksum and may be reused) unless a **hard delete** (GDPR) is requested (Chapter 9.15), which also purges cache rows and Gold embedding rows for those `chunk_id`s and rewrites affected Parquet files.

**Reindex** (model change or corrupted collection):

```mermaid
sequenceDiagram
    participant A as Admin API / DAG
    participant Q as Qdrant
    participant G as Gold
    A->>Q: create collection kb_chunks_v2 (new params)
    A->>G: stream chunk_embeddings for new embedding_version joined with retrieval_units
    loop batches of 512
        A->>Q: upsert(points) into kb_chunks_v2
    end
    A->>Q: count(kb_chunks_v2) == count(v_gold_active_units)?
    A->>Q: update_collection_aliases: kb_chunks -> kb_chunks_v2
    A->>Q: delete kb_chunks_v1 after AKL_REINDEX_GRACE_HOURS
```

The API always queries the alias `kb_chunks`, so cutover is atomic and zero-downtime.

## 5.14 Qdrant Reconciliation Algorithm (used by `akl_qdrant_sync`)

```
gold = SELECT chunk_id, embedded_text_sha256, embedding_version FROM v_gold_active_units JOIN chunk_embeddings USING(chunk_id) WHERE embedding_version = current
qd   = scroll(kb_chunks, with_payload=[chunk_checksum, embedding_version], with_vectors=false)   # paginated, 10k/page
to_upsert = gold − qd (by chunk_id) ∪ {chunk_id ∈ both where payload.embedding_version ≠ current or payload.embedded_text_sha256 ≠ gold}
to_delete = qd − gold (by chunk_id)
upsert in batches of 512 with wait=true; delete in batches of 1000
assert count(qd after) == count(gold)  → else AKL-E5020 and metric akl_qdrant_gold_drift
```

At 100k points the scroll takes ~10 s; at 1B points reconciliation becomes partition-scoped (by `source_type`/`repo`) and change-log driven (Chapter 14).

## 5.15 Failure Handling

| Failure | Handling | Code |
|---|---|---|
| Model download fails | Task retries; model cached on `akl_models` volume after first success | AKL-E5001 |
| ONNX session OOM | Halve batch size, retry up to 3 times | AKL-E5002 |
| Token overflow | Error (chunker invariant broken) | AKL-E5003 |
| NaN/Inf in vector | Drop chunk from batch, flag `embedding_failed` in Postgres, alert | AKL-E5004 |
| Qdrant unavailable | Sync DAG retries; Gold already durable | AKL-E5011 |
| Qdrant partial upsert | Idempotent re-run; upsert by id is a replace | — |
| Drift after sync | Alert; runbook RB-05 | AKL-E5020 |

## 5.16 Folder Structure

```
akl/embedding/
├── __init__.py
├── provider.py          # EmbeddingProvider ABC
├── bge.py               # ONNX/torch BGE implementation, query instruction
├── batcher.py           # length-sorted batching, shard assignment
├── cache.py             # Postgres cache repository
├── writer.py            # gold/chunk_embeddings writer
├── jobs.py              # embedding_jobs bookkeeping, MLflow logging
├── qdrant/
│   ├── client.py        # async/sync client factory, health
│   ├── schema.py        # collection params, payload indexes, alias mgmt
│   ├── reconciler.py    # algorithm 5.14
│   └── reindex.py       # blue/green flow 5.13
└── bm25/
    ├── tokenizer.py     # lowercase, unicode words, identifier splitting (camelCase, snake_case, dots)
    ├── builder.py       # build from gold/retrieval_units, serialize
    └── index.py         # load, search(tokens, filter_fn, k)
```

---

# Chapter 6 — RAG Engine

## 6.1 Overview

The RAG engine (`akl.rag`) converts a user query and conversation state into a cited answer. It is a pipeline of pure, independently testable stages with a shared `RAGContext` object carrying the `request_id`, principal, timings and a `RetrievalTrace` that is persisted for every request.

```mermaid
flowchart LR
    Q[Raw query] --> N[Normalize] --> SC[Spell correct] --> IC[Intent classify] --> EE[Entity extract] --> MI[Metadata infer]
    MI --> QR[Query rewrite<br/>multi-turn resolution]
    QR --> HR[Hybrid retrieve]
    HR --> RR[Rerank]
    RR --> TH{confidence ≥ min?}
    TH -->|no| IE[insufficient_evidence]
    TH -->|yes| CB[Context build]
    CB --> PB[Prompt build]
    PB --> GEN[Generate / stream]
    GEN --> CE[Citation engine]
    CE --> RF[Response format]
    RF --> PM[Persist memory + trace]
```

## 6.2 Query Processing

### 6.2.1 Normalisation

- Unicode NFC; collapse whitespace; strip control chars; lowercase a **copy** for sparse retrieval (dense embedding uses original casing).
- Preserve code-like tokens (`snake_case`, `CamelCase`, `path/like/this`, `ERR_1234`) via a protected-token regex so spell correction and tokenisation do not mangle identifiers.
- Truncate to `AKL_QUERY_MAX_CHARS (2000)`; `AKL-E6001` if empty after normalisation.

### 6.2.2 Spell Correction

`symspellpy` with a dictionary built from the corpus vocabulary (Gold `retrieval_units` term frequencies, rebuilt by the Maintenance DAG into `gold/indexes/vocab/`). Only tokens **absent** from the corpus vocabulary and not protected are corrected (max edit distance 2). Original and corrected queries are both retained; sparse retrieval runs on the corrected form, dense on both (union) when `AKL_QUERY_SPELL_DUAL=true`. Corrections are reported in the trace.

### 6.2.3 Intent Classification

Lightweight, rules + tiny classifier (logistic regression over bge query embedding, trained on synthetic labelled set shipped in `configs/intents.yaml`):

| Intent | Example | Effect |
|---|---|---|
| `factual_lookup` | "what port does MinIO use" | Standard hybrid, k=8 |
| `how_to` | "how do I configure retries" | Boost `chunk_type in (prose, code)`; k=10 |
| `code_search` | "function that builds the BM25 index" | Boost `chunk_type=code`, sparse weight ↑ |
| `troubleshooting` | "AKL-E5020 drift error" | Protected tokens → exact-match sparse boost; runbook boost |
| `comparison` | "difference between Silver and Gold" | Retrieve for each entity separately then merge |
| `summarization` | "summarize the security chapter" | Section-level (parent) retrieval when enabled; larger context budget |
| `chitchat` | "thanks" | No retrieval; canned response |
| `out_of_scope` | "weather in Paris" | Refuse with `out_of_scope` |

### 6.2.4 Entity Extraction

Regex + gazetteer built from corpus metadata: repo names, file paths, error codes (`AKL-E\d{4}`), env vars (`AKL_[A-Z_]+`), service names, version strings, dates. Entities become (a) exact-match `must` terms for sparse retrieval and (b) candidate metadata filters.

### 6.2.5 Metadata Inference

| Signal | Inferred filter |
|---|---|
| Repo name entity | `repo = X` (soft: applied as boost first; hard only if user filter or explicit "in repo X") |
| "pdf", "spec", "the whitepaper" | `source_type = pdf` (soft) |
| Language keyword ("in Python", "yaml") | `code_language = X` (soft) |
| Temporal ("latest", "recent", "as of 2026") | `document_updated_at ≥ now − 90d` (soft) |
| User-provided `filters` in request | hard |
| Principal's `allowed_security_levels` / groups | **always hard** |

Soft filters are implemented as a two-pass retrieval: filtered first; if `< k/2` results, unfiltered second pass merged with a 0.85 score multiplier.

### 6.2.6 Multi-turn Query Rewrite

If `conversation_id` is present and the query contains anaphora (`it`, `that`, `this one`, `the second option`) or is < 4 tokens, the LLM (or a rule-based fallback substituting the last mentioned entity) rewrites the query into a standalone form using the conversation summary + last 3 turns. The rewritten query is used for retrieval; the original is shown to the model. Trace stores both.

## 6.3 Hybrid Retrieval

### 6.3.1 Dense Retrieval

`query_vec = embed_query("Represent this sentence for searching relevant passages: " + query)`; Qdrant `search(collection="kb_chunks", vector, limit=AKL_RETRIEVAL_DENSE_K (50), query_filter=security ∧ hard filters, search_params={hnsw_ef: AKL_QDRANT_HNSW_EF (128), exact: false}, with_payload=true)`.

### 6.3.2 BM25 (Sparse) Retrieval

Index built from Gold with tokenizer: lowercase → Unicode word boundaries → split identifiers on `_`, `.`, `/`, camelCase humps (emitting both the whole identifier and its parts) → drop stopwords except when query is short (`≤ 3 tokens`, stopwords retained) → no stemming for identifiers, Snowball stemming for prose tokens. BM25 parameters `k1=1.5`, `b=0.75`. Filtering: BM25 returns top-`AKL_RETRIEVAL_SPARSE_K × 4` then applies the same security/hard filters in memory (payload held alongside index) and truncates to `sparse_k` (50). Exact entity terms use a boolean `must` pre-filter over an inverted index of protected tokens.

### 6.3.3 Reciprocal Rank Fusion

```
for list in [dense_results, sparse_results]:
    for rank, cand in enumerate(list, start=1):
        fused[cand.chunk_id] += weight_list / (AKL_RRF_K (60) + rank)
weight_dense  = 1.0 ; weight_sparse = 1.0 (code_search/troubleshooting: 1.3)
fused[c] *= 0.9 if c.quality_flag == "marginal"
fused[c] *= 1.05 if c matches soft filter
take top AKL_RETRIEVAL_FUSED_K (40)
```

RRF is used because BM25 and cosine scores are not commensurable; rank-based fusion needs no calibration and is robust across corpora.

### 6.3.4 Cross-Encoder Reranking

Model `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, ONNX). Input pairs `(query, context_prefix + text)`, truncated to 512 tokens. Batch 40 pairs ≈ 120–250 ms CPU. Output logits → sigmoid → `rerank_score ∈ (0,1)`. Final ranking by `rerank_score`; RRF score retained in trace. Reranking can be disabled per request (`rerank=false`) or globally (`AKL_RERANK_ENABLED`).

### 6.3.5 Confidence Threshold

`confidence = rerank_score[0]` (top result). If `confidence < AKL_RAG_MIN_CONFIDENCE (0.35)` → `insufficient_evidence`. Secondary guard: if fewer than `AKL_RAG_MIN_CANDIDATES (2)` candidates exceed `0.20`, also refuse. Thresholds are calibrated per model via the evaluation DAG (Chapter 12.7), stored in MLflow, and can be updated via Admin API without redeploy.

## 6.4 Context Builder

1. Take top `AKL_RAG_TOP_K (8)` reranked candidates.
2. **Deduplicate**: drop candidate B if `jaccard(3-grams(A), 3-grams(B)) ≥ 0.85` with a higher-ranked A; drop overlap text between adjacent chunks (`prev/next` linkage + `overlap_prev_tokens`).
3. **Neighbour expansion** (optional): fetch `prev/next` for top-2 when `AKL_RAG_NEIGHBOR_EXPANSION ≥ 1`.
4. **Ordering**: group by `document_id`; within document by `chunk_index`; documents ordered by best rerank score. Rationale: models read coherent document order better than interleaved snippets; the top document still appears first.
5. **Budget**: `AKL_RAG_CONTEXT_TOKENS (3000)` for context blocks measured with the generation model tokenizer (`tiktoken cl100k_base` as a proxy when unknown). Truncate lowest-ranked whole chunks first; never truncate mid-chunk except the last one, which is cut at a sentence boundary.
6. Render each block as:

```
[n] source=<source_type> title="<title>" locator="<citation locator>" chunk_id=<uuid>
<context_prefix>
<text>
```

## 6.5 Prompt Builder and Template

System prompt (`configs/prompts/answer_v1.md`, version-controlled; `prompt_version` recorded in trace):

```
You are the internal knowledge assistant for <ORG_NAME>. Answer ONLY from the numbered context blocks.
Rules:
1. Every factual sentence must end with one or more citation markers like [2] or [1][4] referring to block numbers.
2. If the context does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE and explain in one sentence what is missing.
3. Never invent file paths, commands, versions, or numbers that do not appear in the context.
4. Prefer the most recent document when blocks conflict, and say that they conflict.
5. Quote code and configuration verbatim from the context in fenced blocks.
6. Be concise: ≤ 250 words unless the question asks for a list or procedure.
```

Message structure: `system` → `conversation summary (if any)` → last `AKL_RAG_HISTORY_TURNS (3)` turns → `user: <context blocks>\n\nQuestion: <original query>`. Total budget `AKL_LLM_MAX_INPUT_TOKENS (6000)`.

## 6.6 Hallucination Prevention

| Mechanism | Description |
|---|---|
| Closed-book prohibition | System rule 1–3 |
| Confidence gate | Chapter 6.3.5 refuses before generation |
| Citation validation | Answer sentences without a valid `[n]` marker are flagged; if `> AKL_RAG_MAX_UNCITED_RATIO (0.2)` the response is marked `low_faithfulness` and, when `AKL_RAG_STRICT=true`, replaced with the extractive answer |
| Marker validation | `[n]` outside `1..len(blocks)` → removed and flagged |
| Numeric/identifier check | Numbers, paths, env vars, error codes in the answer must appear in cited blocks (string containment); violations flagged `unsupported_token` |
| INSUFFICIENT_EVIDENCE passthrough | Model-declared insufficiency mapped to `reason=model_declared_insufficient` |
| Temperature | `AKL_LLM_TEMPERATURE=0.1` |
| Post-hoc NLI (optional) | `cross-encoder/nli-deberta-v3-small` entailment of each sentence vs its cited block; below 0.5 → flag (Enterprise Scale default on) |

## 6.7 Citation Engine

1. Parse `[n]`, `[n][m]`, `[n, m]` markers with regex; map `n → block → chunk_id, document_id, locator`.
2. Build `citations[]` ordered by first appearance: `{index, chunk_id, lineage_id, document_id, title, source_type, locator, url, snippet (first 200 chars), rerank_score}`.
3. Rewrite markers in the final text to sequential `[1..k]` after dropping invalid ones.
4. Store `answer_citations` rows (Appendix A) for audit and for offline faithfulness evaluation.
5. Guarantee (G5): if zero valid citations remain and mode is generative → convert to extractive answer (which always cites).

## 6.8 Conversation Memory and Summarisation

- Tables `conversations` and `messages` (Appendix A). Each turn stores `role`, `content`, `rewritten_query`, `citations`, `trace_id`, `token_count`.
- Working memory = `summary` + last `AKL_RAG_HISTORY_TURNS` turns.
- When the un-summarised turns exceed `AKL_RAG_SUMMARY_TRIGGER_TOKENS (1500)`, an asynchronous background task (FastAPI `BackgroundTasks`) calls the LLM with `prompts/summarize_v1.md` to fold older turns into `summary` (≤ 300 tokens), preserving entities, decisions and cited document titles. Without an LLM, summary = concatenated rewritten queries (rule-based).
- Conversations expire after `AKL_CONVERSATION_TTL_DAYS (30)` (Maintenance DAG).

## 6.9 Streaming Responses

- Endpoint `POST /v1/chat` with `stream=true` returns `text/event-stream`.
- Events: `meta` (request_id, trace_id, retrieval summary) → `token` (repeated) → `citations` → `done` (confidence, flags, timings) or `error`.
- Server-side: `httpx.AsyncClient.stream` against the LLM; tokens forwarded as received; citation parsing performed on the accumulated text at the end. Partial `[n` markers are buffered until closed to avoid emitting broken markers.
- Non-streaming path returns the same fields in one JSON body.

## 6.10 Multi-turn Retrieval

Retrieval on turn *t* uses the rewritten query. Additionally, chunks cited in the previous turn are **re-injected as candidates** with RRF bonus `+0.01` (keeps referents stable when the user says "explain step 3 more"). Citations reference `lineage_id` so that if the document was re-chunked between turns the follow-up still resolves.

## 6.11 Response Formatter

```json
{
  "request_id": "…", "conversation_id": "…", "trace_id": "…",
  "mode": "generative | extractive",
  "answer": "…markdown with [1] markers…",
  "citations": [ { "index": 1, "chunk_id": "…", "document_id": "…", "title": "…", "source_type": "github", "locator": "repo/path#L10-L42 @ main", "url": "https://…", "snippet": "…", "score": 0.83 } ],
  "confidence": 0.83,
  "flags": ["low_faithfulness"],
  "retrieval": { "dense_k": 50, "sparse_k": 50, "fused_k": 40, "reranked": true, "filters": {…}, "rewritten_query": "…", "intent": "how_to" },
  "timings_ms": { "query_processing": 12, "embed_query": 18, "dense": 35, "sparse": 9, "rerank": 190, "context": 4, "llm_first_token": 620, "llm_total": 2400, "total": 2710 }
}
```

## 6.12 Failure Handling

| Failure | Behaviour | Code |
|---|---|---|
| Qdrant down | Sparse-only retrieval, flag `dense_unavailable`; 200 with degraded flag | AKL-W6010 |
| BM25 index not loaded | Dense-only, flag `sparse_unavailable` | AKL-W6011 |
| Both unavailable | 503 | AKL-E6012 |
| Reranker error | Skip rerank, use RRF order, flag | AKL-W6013 |
| LLM timeout (`AKL_LLM_TIMEOUT_S=60`) or 5xx | Extractive fallback, flag `llm_unavailable` | AKL-W6020 |
| Embedding service error | 503 (dense) → sparse-only if possible | AKL-W6014 |
| Context overflow | Truncate per 6.4 step 5 | — |
| Conversation not found / not owned | 404 / 403 | AKL-E6030 / AKL-E6031 |

## 6.13 Latency Optimisation

- Warm models at startup (`lifespan`): tokenizer, BGE ONNX session, reranker session, BM25 index, spell dictionary. Readiness probe waits for them.
- Dense and sparse searches run concurrently (`asyncio.gather`).
- Qdrant gRPC with connection reuse; `hnsw_ef` adaptive: 64 for `k ≤ 5`, 128 default, 256 when `precision=high`.
- Reranker batch executed in a thread-pool executor (`AKL_RERANK_THREADS`); ONNX INT8.
- Query embedding LRU cache (`AKL_QUERY_CACHE_SIZE=2048`, keyed by normalised query) — repeated questions skip embedding.
- Result cache (optional, `AKL_RESULT_CACHE_TTL_S=0` off by default) keyed by `(normalized_query, filters, principal_groups, gold_snapshot_id)`.
- Streaming reduces perceived latency; first token target ≤ 2.5 s.

## 6.14 Retrieval Lifecycle Sequence

```mermaid
sequenceDiagram
    participant API
    participant QP as QueryProcessor
    participant EMB as Embedder
    participant QD as Qdrant
    participant BM as BM25
    participant F as Fusion
    participant RR as Reranker
    API->>QP: process(query, principal, filters, memory)
    QP-->>API: ProcessedQuery(rewritten, tokens, entities, hard/soft filters, intent)
    par
        API->>EMB: embed_query
        EMB-->>API: vec
        API->>QD: search(vec, filter, 50)
        QD-->>API: dense[50]
    and
        API->>BM: search(tokens, filter, 50)
        BM-->>API: sparse[50]
    end
    API->>F: rrf(dense, sparse, weights)
    F-->>API: fused[40]
    API->>RR: score(query, fused)
    RR-->>API: ranked[40] + confidence
```

## 6.15 RAG Folder Structure

```
akl/rag/
├── __init__.py
├── context.py           # RAGContext, RetrievalTrace
├── query/
│   ├── normalize.py
│   ├── spell.py
│   ├── intent.py
│   ├── entities.py
│   ├── filters.py       # metadata inference, principal filter injection
│   └── rewrite.py       # multi-turn rewrite
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   ├── fusion.py
│   ├── rerank.py
│   └── engine.py        # HybridRetriever
├── context_builder.py
├── prompt.py            # template loading, versioning
├── llm/
│   ├── provider.py      # LLMProvider ABC, stream/complete
│   ├── openai_compat.py
│   └── extractive.py    # fallback provider
├── citations.py
├── memory.py            # conversations, summarisation task
├── guards.py            # hallucination checks
├── formatter.py
└── service.py           # RAGService.answer(...), .stream(...)
```

---

# Chapter 7 — Airflow Orchestration

## 7.1 Airflow Fundamentals as Used by AKL

| Component | Role in AKL | MVP configuration |
|---|---|---|
| Scheduler | Parses DAG files every 30 s, evaluates schedules and dependencies, queues task instances | 1 container, `LocalExecutor`, `parallelism=8`, `max_active_runs_per_dag=1` |
| Executor | Runs task instances | LocalExecutor (subprocesses within scheduler container). A dedicated `airflow-worker` container is present and idle unless `CeleryExecutor` is enabled (ADR-007) |
| Metadata DB | Task states, XComs, variables, connections | Postgres database `airflow` |
| Webserver | UI + REST API (used by FastAPI to trigger runs) | 1 container, port 8080 |
| Triggerer | Runs deferrable sensors | 1 container (lightweight) |
| Sensors | Wait for conditions (Bronze backlog present, Qdrant healthy) | Deferrable `PythonSensor` wrappers with `mode="reschedule"` |
| Operators | AKL uses `@task` (TaskFlow) Python operators wrapping `akl.*` services; no BashOperators for logic | |
| XCom | Small JSON payloads only: counts, lists of `run_id`s, `gold_snapshot_id`. Never rows. Large lists → Postgres tables keyed by `run_id` | `AIRFLOW__CORE__MAX_XCOM_SIZE` default |
| Trigger rules | `all_success` default; `none_failed_min_one_success` for fan-in after dynamic mapping; `all_done` for `emit_metrics` and `finalize_run` | |
| Datasets (data-aware scheduling) | Downstream DAGs triggered by upstream dataset updates (`Dataset("akl://silver/documents")` etc.) in addition to cron | |
| Pools | `akl_embedding` (slots=`AKL_EMBED_TASK_SHARDS`), `akl_github_api` (slots=2) | |

All DAGs share `default_args`: `owner="akl"`, `retries=2`, `retry_delay=2 min`, `retry_exponential_backoff=True`, `max_retry_delay=20 min`, `execution_timeout` per task, `on_failure_callback=akl.airflow.callbacks.on_failure` (emits metric, structured log, optional Slack), `sla_miss_callback`.

Every task function is a thin wrapper: build `RunContext(run_id, task_id, correlation_id, try_number)`, open OTel span, call service, log JSON summary, push small XCom.

## 7.2 DAG Inventory

| DAG ID | Schedule | Purpose | Triggers next |
|---|---|---|---|
| `akl_ingestion` | `*/30 * * * *` + manual + API trigger | Discover/fetch/parse/validate to Silver documents | `akl_chunking` via Dataset |
| `akl_chunking` | Dataset-triggered + `15 * * * *` safety cron | Chunk current documents lacking chunks | `akl_embedding` |
| `akl_embedding` | Dataset-triggered + `30 * * * *` | Embed backlog, write Gold | `akl_qdrant_sync` |
| `akl_qdrant_sync` | Dataset-triggered + `45 * * * *` | Reconcile Qdrant + rebuild BM25 | — |
| `akl_maintenance` | `0 2 * * *` daily | Compaction, retention, cache eviction, stats, eval, backups | — |

## 7.3 DAG 1 — `akl_ingestion`

**Purpose**: Bring all configured sources to Bronze and Silver incrementally.

**Trigger**: cron every 30 min; manual; `POST /v1/admin/pipelines/ingestion/trigger` with `conf={"connectors": [...], "document_ids": [...]}`.

```mermaid
flowchart LR
    S[start] --> LC[load_connector_configs]
    LC --> D1[discover<br/>dynamic map per connector]
    D1 --> F1[fetch_to_bronze<br/>dynamic map per connector]
    F1 --> CK[checkpoint_connectors]
    CK --> LB[list_bronze_backlog]
    LB --> PS[parse_validate_to_silver<br/>dynamic map over N shards]
    PS --> DD[dedup_pass]
    DD --> QG{quality_gate}
    QG -->|pass| PUB[publish_silver_dataset]
    QG -->|fail| AL[alert_and_hold]
    PUB --> EM[emit_metrics]
    AL --> EM
    EM --> E[end]
```

| Task | Timeout | Retries | Notes |
|---|---|---|---|
| `load_connector_configs` | 1 m | 1 | Reads `configs/connectors/*.yaml`, validates, returns list |
| `discover[c]` | 15 m | 2 | Pool `akl_github_api` for GitHub |
| `fetch_to_bronze[c]` | 45 m | 2 | Concurrency inside task: `AKL_FETCH_CONCURRENCY (8)` |
| `checkpoint_connectors` | 2 m | 2 | Only after all fetches; `trigger_rule=none_failed_min_one_success` |
| `list_bronze_backlog` | 5 m | 2 | Postgres query `status='bronze'`; shards by `hash(document_id) % AKL_PARSE_SHARDS (4)` |
| `parse_validate_to_silver[s]` | 60 m | 1 | Parser subprocess isolation |
| `dedup_pass` | 15 m | 2 | SimHash vs ledger |
| `quality_gate` | 2 m | 0 | Fails run if quarantine ratio > `AKL_GATE_QUARANTINE_RATIO (0.25)` or zero docs parsed while backlog > 0 |
| `publish_silver_dataset` | 1 m | 2 | `outlets=[Dataset("akl://silver/documents")]` |
| `emit_metrics` | 2 m | 0 | `trigger_rule=all_done` |

**SLA**: run completes within 25 min. **Backfill**: `catchup=False`; a manual backfill re-runs with `conf.force_reparse=true` which ignores the `(sha, parser_version)` skip for listed documents. **Idempotency**: Bronze content-addressed; Silver keyed by `document_version_id`; re-run writes zero new rows for unchanged inputs. **Metrics**: `akl_ingestion_documents_discovered_total`, `_fetched_total`, `_dedup_hits_total`, `_parsed_total`, `_quarantined_total{code}`, `akl_ingestion_run_duration_seconds`, `akl_bronze_bytes_written_total`.

**Failure recovery**: fetch failures do not advance checkpoint → next run retries; parse failures quarantine (run still succeeds unless gate); Postgres failure fails task, safe to retry.

## 7.4 DAG 2 — `akl_chunking`

```mermaid
flowchart LR
    S[start] --> LB[list_documents_needing_chunks]
    LB --> CH[chunk_documents<br/>dynamic map over shards]
    CH --> AG[aggregate_chunk_diffs]
    AG --> QG{quality_gate}
    QG -->|pass| PUB[publish_silver_chunks]
    QG -->|fail| AL[alert]
    PUB --> GU[refresh_gold_retrieval_units]
    GU --> EM[emit_metrics]
```

- `list_documents_needing_chunks`: `v_current_documents` LEFT JOIN chunks on `(document_version_id, chunker_version, chunk_config_hash)` where NULL, plus `conf.document_ids`.
- `chunk_documents[s]`: runs incremental algorithm 4.12; timeout 45 m.
- `quality_gate`: fail if mean chunk quality drops > 0.1 vs previous snapshot or if `low_quality` ratio > 0.3.
- `refresh_gold_retrieval_units`: DuckDB SQL `sql/gold/retrieval_units.sql` — projection of `v_gold_active_units`, writes changed partitions only (partitions touched by this run's documents), stamps `gold_snapshot_id = run_id`.
- Schedule: Dataset `akl://silver/documents` + hourly cron. SLA 30 m. Metrics: `akl_chunks_created_total{status}`, `akl_chunk_tokens` histogram, `akl_chunk_quality` histogram.

## 7.5 DAG 3 — `akl_embedding`

```mermaid
flowchart LR
    S[start] --> WM[warm_model_check]
    WM --> LB[compute_embedding_backlog]
    LB --> SH[shard_backlog]
    SH --> EB[embed_shard<br/>dynamic map, pool akl_embedding]
    EB --> MG[merge_and_write_gold]
    MG --> ML[log_mlflow_run]
    ML --> QG{coverage_gate}
    QG -->|pass| PUB[publish_gold_embeddings]
    QG -->|fail| AL[alert]
    PUB --> EM[emit_metrics]
```

- `warm_model_check`: model files present on `akl_models` volume; download if not; checks dim==384.
- `compute_embedding_backlog`: `v_embedding_coverage` where vector IS NULL → writes backlog to Postgres `embedding_backlog(run_id, chunk_id, shard)`.
- `embed_shard[s]`: cache lookup → generate → write Gold part file for shard; timeout 90 m; retries 2 with batch halving on OOM.
- `coverage_gate`: fail if coverage < 0.99 of active units (except `embedding_failed`).
- MLflow run params: model, version, batch, threads, INT8; metrics: `chunks_total, cache_hit_rate, throughput_cps, mean_norm, p50_latency_ms`.
- Metrics: `akl_embeddings_generated_total`, `akl_embedding_cache_hits_total`, `akl_embedding_batch_seconds`, `akl_embedding_backlog_size`.

## 7.6 DAG 4 — `akl_qdrant_sync`

```mermaid
flowchart LR
    S[start] --> HC[qdrant_health_sensor]
    HC --> EC[ensure_collection_and_indexes]
    EC --> DF[compute_diff]
    DF --> UP[upsert_points<br/>batches]
    DF --> DL[delete_points]
    UP & DL --> VF[verify_counts]
    VF --> BM[rebuild_bm25_index]
    BM --> VC[rebuild_vocab]
    VC --> NT[notify_api_reload]
    NT --> EM[emit_metrics]
```

- `qdrant_health_sensor`: deferrable, poke 30 s, timeout 10 m.
- `ensure_collection_and_indexes`: idempotent create; verifies payload indexes; creates alias if absent.
- `compute_diff`: algorithm 5.14; writes `to_upsert/to_delete` to Postgres `qdrant_sync_ops(run_id, op, chunk_id)`.
- `upsert_points`: batches 512, `wait=true`, retries 3 with backoff; `delete_points` batches 1000.
- `verify_counts`: `count(qdrant) == count(gold active)` else `AKL-E5020` (task fails, run marked failed, alert; upserts are idempotent so retry is safe).
- `rebuild_bm25_index`: full rebuild from Gold (fast; incremental BM25 deferred to Enterprise Scale) → `gold/indexes/bm25/version=<gold_snapshot_id>/`; keeps last 3 versions.
- `notify_api_reload`: `POST /v1/admin/reload-index` with service token; API hot-swaps BM25 and vocab.
- Metrics: `akl_qdrant_points_upserted_total`, `_deleted_total`, `akl_qdrant_gold_drift`, `akl_qdrant_sync_duration_seconds`, `akl_bm25_build_seconds`, `akl_bm25_index_terms`.

## 7.7 DAG 5 — `akl_maintenance`

```mermaid
flowchart LR
    S[start] --> C[compact_partitions]
    S --> R1[bronze_retention]
    S --> R2[quarantine_retention]
    S --> R3[embedding_cache_eviction]
    S --> R4[conversation_ttl]
    S --> R5[retire_old_embedding_versions]
    C --> ST[compute_corpus_stats]
    ST --> EV[run_retrieval_evaluation]
    EV --> ML[log_eval_to_mlflow]
    R1 & R2 & R3 & R4 & R5 & ML --> BK[backup_postgres_and_qdrant_snapshot]
    BK --> VC[vacuum_analyze_postgres]
    VC --> EM[emit_metrics]
```

- Daily 02:00. Each retention task is idempotent (deletes only what matches policy). `run_retrieval_evaluation` executes Chapter 12.7 against `gold/eval/qa_pairs` current version and writes `gold/eval/results`. `backup_*`: `pg_dump` of `akl` DB to `s3://akl-lakehouse/backups/postgres/<date>.sql.gz`; Qdrant `POST /collections/kb_chunks_v1/snapshots` then copy to `backups/qdrant/`. Retention of backups `AKL_BACKUP_RETENTION_DAYS (14)`.
- SLA: 90 m. Metrics: `akl_compaction_*`, `akl_retention_deleted_total{dataset}`, `akl_eval_recall_at_10`, `akl_eval_mrr`, `akl_backup_bytes`.

## 7.8 Cross-DAG Data Quality Gates Summary

| Gate | DAG | Condition to fail | Action |
|---|---|---|---|
| Quarantine ratio | ingestion | quarantined / fetched > 0.25 | Fail run, alert, hold downstream (no Dataset publish) |
| Zero-parse anomaly | ingestion | backlog > 0 ∧ parsed = 0 | Fail |
| Chunk quality regression | chunking | mean quality drop > 0.1 | Fail |
| Embedding coverage | embedding | coverage < 0.99 | Fail |
| Qdrant drift | sync | drift ≠ 0 | Fail |
| Eval regression | maintenance | recall@10 drop > 0.05 vs 7-day median | Warn + alert (non-blocking) |

## 7.9 Airflow Folder Structure and Configs

```
airflow/
├── dags/
│   ├── akl_ingestion.py
│   ├── akl_chunking.py
│   ├── akl_embedding.py
│   ├── akl_qdrant_sync.py
│   └── akl_maintenance.py
├── plugins/
│   └── akl_airflow/
│       ├── callbacks.py       # on_failure, sla_miss, metrics push (Pushgateway or StatsD)
│       ├── context.py         # RunContext from Airflow context
│       ├── sensors.py         # QdrantHealthSensor, BronzeBacklogSensor
│       └── datasets.py        # Dataset URIs
├── config/
│   ├── airflow.cfg.tmpl       # rendered from env at container start
│   └── pools.yaml             # akl_embedding, akl_github_api
└── requirements-airflow.txt
configs/
├── dags/
│   ├── ingestion.yaml         # schedule, shards, gate thresholds
│   ├── chunking.yaml
│   ├── embedding.yaml
│   ├── qdrant_sync.yaml
│   └── maintenance.yaml
└── connectors/
    ├── github__akl-docs.yaml
    ├── pdf__inbox.yaml
    ├── markdown__runbooks.yaml
    └── html__internal-wiki.yaml
```

DAG YAML fields: `schedule`, `max_active_runs`, `default_retries`, `retry_delay_s`, `execution_timeout_s` per task, `sla_minutes`, `shards`, `gates: {…}`, `pools`. DAG Python files read their YAML at parse time so operational tuning is config-only.

---

# Chapter 8 — Observability & DataOps

## 8.1 Principles

1. Every unit of work (HTTP request, DAG task, batch) has a `correlation_id`.
2. Logs are structured JSON, never free text; every log line is machine-parseable.
3. Metrics are the primary alerting surface; logs are for diagnosis; traces are for latency attribution.
4. Data quality is observable in the same stack as system health (freshness, drift, coverage as Prometheus gauges).
5. Every alert has a runbook.

## 8.2 Structured Logging

Library: `structlog` with JSON renderer; stdlib logging bridged. Output to stdout (collected by Docker; optional Loki via Promtail in `docker-compose.observability.yml`).

### 8.2.1 Log Schema

| Field | Type | Always | Description |
|---|---|---|---|
| `ts` | RFC3339 ns | yes | |
| `level` | string | yes | `debug/info/warning/error/critical` |
| `service` | string | yes | `akl-api`, `akl-airflow`, `akl-embedding` |
| `component` | string | yes | Python module path |
| `event` | string | yes | snake_case event name, e.g. `bronze_object_written` |
| `correlation_id` | string | yes | `request_id` or `run_id/task_id/try` |
| `request_id` | string | API | |
| `trace_id`, `span_id` | hex | when tracing | OTel IDs for log↔trace linking |
| `principal_id` | string | API | hashed subject |
| `dag_id`,`task_id`,`run_id`,`try_number` | | Airflow | |
| `document_id`,`chunk_id` | UUID | when applicable | |
| `duration_ms` | int | timed events | |
| `error_code` | string | errors | `AKL-Exxxx` |
| `error_type`, `stack_hash` | string | errors | Exception class, sha1 of traceback frames (for grouping) |
| `extra` | object | optional | Bounded to 2 KB |

Rules: never log document text, secrets, tokens, or full stack traces at `info`; PII fields redacted by a `structlog` processor using the same regex set as the ingestion secret scanner. Log level via `AKL_LOG_LEVEL`. Sampling of `debug` logs at 1% in API when `AKL_LOG_SAMPLE_DEBUG=true`.

### 8.2.2 Request and Correlation IDs

- API middleware reads `X-Request-ID` (validated UUIDv4) or generates one; echoes it in the response header; binds it to `structlog` contextvars and to the OTel span attribute `akl.request_id`.
- When the API triggers an Airflow run, `conf.correlation_id = request_id`; tasks log both their `run_id` and the originating `correlation_id`, giving end-to-end lineage from an upload to its Qdrant points.

## 8.3 Prometheus Metrics

Exposition: FastAPI via `prometheus-fastapi-instrumentator` + custom registry at `/metrics`; Airflow tasks push to **Pushgateway** (`akl-pushgateway:9091`, part of Compose) with grouping key `dag_id/task_id`, because short-lived tasks cannot be scraped; long-running embedding shards additionally expose a per-process gauge. Airflow's own StatsD metrics are exported via `statsd-exporter` for scheduler health. The full catalog is Appendix F; the categories:

| Category | Examples |
|---|---|
| API | `akl_http_requests_total{method,route,status}`, `akl_http_request_duration_seconds` |
| Retrieval | `akl_search_latency_seconds{stage}`, `akl_retrieval_candidates{stage}`, `akl_rerank_confidence`, `akl_insufficient_evidence_total`, `akl_result_cache_hits_total` |
| RAG | `akl_llm_latency_seconds{phase}`, `akl_llm_tokens_total{direction}`, `akl_answer_citations`, `akl_answer_flags_total{flag}`, `akl_llm_cost_usd_total` |
| Ingestion | per DAG (Chapter 7) |
| Chunking | `akl_chunk_tokens`, `akl_chunk_quality`, `akl_chunks_created_total{status}` |
| Embedding | `akl_embedding_*` |
| Vector | `akl_qdrant_points`, `akl_qdrant_gold_drift`, `akl_qdrant_request_duration_seconds` |
| Freshness | `akl_data_freshness_seconds{source_type}`, `akl_gold_snapshot_age_seconds` |
| Lakehouse | `akl_lakehouse_partition_files{dataset}`, `akl_lakehouse_bytes{layer}`, `akl_compaction_*` |
| Pipeline | `akl_dag_run_total{dag_id,state}`, `akl_task_duration_seconds{dag_id,task_id}`, `akl_sla_miss_total` |
| Quality/Eval | `akl_eval_recall_at_10`, `akl_eval_mrr`, `akl_eval_faithfulness` |

Histogram buckets: latency `[.005,.01,.025,.05,.1,.25,.5,1,2.5,5,10,30]`; tokens `[32,64,128,192,256,320,384,448,512]`.

## 8.4 OpenTelemetry Tracing

- SDK in API and in Airflow task wrapper; exporter OTLP gRPC to `akl-otel-collector:4317`; collector exports to Grafana Tempo (optional profile) or to a debug logging exporter by default.
- Span hierarchy (chat): `http POST /v1/chat` → `rag.query_process` → `rag.retrieve` → [`embed.query`, `qdrant.search`, `bm25.search`] → `rag.fuse` → `rag.rerank` → `rag.context` → `llm.stream` → `rag.citations` → `db.persist`.
- Span hierarchy (pipeline): `airflow.task <dag>.<task>` → `connector.fetch` (per item) → `s3.put` → `db.upsert` … .
- Attributes: `akl.request_id`, `akl.document_id`, `akl.chunk_count`, `akl.batch_size`, `akl.cache_hit`.
- Sampling: API `parentbased_traceidratio` 0.2 (`AKL_OTEL_SAMPLE_RATIO`); pipelines 1.0.

## 8.5 Pipeline Lineage

Table `lineage_edges(run_id, task_id, input_dataset, input_partition, output_dataset, output_partition, rows_in, rows_out, created_at)` written by every task that reads/writes a Lakehouse dataset (via the `LakehouseIO` wrapper). This yields a queryable lineage graph: "which Qdrant points derive from Bronze object sha X?" = manifest → document_version → chunks → embeddings → sync ops. Admin API `GET /v1/admin/lineage/document/{id}` renders it. Enterprise Scale: emit OpenLineage events to Marquez/DataHub.

## 8.6 Grafana Dashboards (provisioned from `observability/grafana/dashboards/*.json`)

| Dashboard | Panels |
|---|---|
| **API Overview** | RPS, error rate, p50/p95/p99 latency by route, active streams, rate-limit rejections, auth failures |
| **Retrieval Quality** | confidence distribution, insufficient-evidence rate, dense vs sparse contribution (fraction of final top-8 sourced from each), rerank latency, result-cache hit rate |
| **RAG Generation** | LLM first-token/total latency, tokens in/out, cost/day, answer flags rate, citation count distribution, extractive fallback rate |
| **Pipelines** | DAG run states over time, task durations heatmap, SLA misses, retries, quarantine by code |
| **Corpus Health** | documents/chunks per source, token histogram, quality distribution, duplicate rate, embedding coverage, backlog size, Qdrant point count vs Gold, drift |
| **Freshness SLA** | per-source freshness gauge vs SLO, Gold snapshot age, time since last successful sync |
| **Infrastructure** | container CPU/mem (cAdvisor optional), Postgres connections, MinIO bucket bytes by prefix, Qdrant RAM, disk free |
| **Cost** | estimated compute-hours per DAG, LLM spend, storage bytes × unit price (config) |

## 8.7 MLflow Experiment Tracking

| Experiment | Runs logged by | Params | Metrics | Artefacts |
|---|---|---|---|---|
| `akl/embedding` | `akl_embedding` DAG | model, version, batch, device, int8, shards | throughput, cache hit rate, norms, failures | backlog summary CSV |
| `akl/retrieval-eval` | `akl_maintenance` and CI | k values, rrf_k, rerank on/off, hnsw_ef, gold_snapshot_id, eval version | recall@k, MRR, nDCG@10, latency p50/p95 | per-query results Parquet, confusion of intents |
| `akl/rag-eval` | maintenance (if LLM) | prompt_version, temperature, context tokens | faithfulness, citation coverage, answer relevance, refusal rate | judged samples JSONL |
| `akl/chunking` | manual experiments | config hash, target/overlap | mean quality, tokens, chunk count | config YAML |

Model registry: `akl-embedder` (bge-small, stage Production) and `akl-reranker`; the API reads the Production version's `embedding_version` tag at startup and refuses to start if it does not match Qdrant alias payload version (`AKL-E8001`).

## 8.8 Data Freshness and SLA Metrics

`akl_data_freshness_seconds{source_type}` = `now − max(document_updated_at)` over Gold active units, computed by `akl_qdrant_sync.emit_metrics` and by an API background task every 60 s. SLOs: github 3600 s, html 21600 s, pdf/markdown n/a (event-driven). `akl_pipeline_last_success_timestamp{dag_id}` drives "time since success" panels.

## 8.9 Alert Rules (`observability/prometheus/alerts.yml`)

| Alert | Expression (summary) | Severity | Runbook |
|---|---|---|---|
| `AKLApiHighErrorRate` | 5xx ratio > 2% for 5 m | critical | RB-01 |
| `AKLApiLatencyP95` | search p95 > 0.8 s for 10 m | warning | RB-02 |
| `AKLDagFailed` | `increase(akl_dag_run_total{state="failed"}[1h]) > 0` | warning | RB-03 |
| `AKLSlaMiss` | `akl_sla_miss_total` increases | warning | RB-03 |
| `AKLFreshnessBreached` | `akl_data_freshness_seconds{source_type="github"} > 3600` for 15 m | warning | RB-04 |
| `AKLQdrantDrift` | `akl_qdrant_gold_drift != 0` for 30 m | critical | RB-05 |
| `AKLEmbeddingBacklogGrowing` | backlog gauge increasing for 3 h | warning | RB-06 |
| `AKLQuarantineSpike` | quarantine ratio > 0.25 in a run | warning | RB-07 |
| `AKLInsufficientEvidenceSpike` | rate > 0.4 of chats for 30 m | warning | RB-08 |
| `AKLEvalRegression` | recall@10 < 7-day median − 0.05 | warning | RB-09 |
| `AKLDiskLow` | MinIO/Postgres volume free < 15% | critical | RB-10 |
| `AKLServiceDown` | `up == 0` for 2 m | critical | RB-01 |

Alertmanager routes: `critical` → Slack `#akl-alerts` + email; `warning` → Slack. Webhook URL / SMTP via `AKL_ALERT_SLACK_WEBHOOK`, `AKL_ALERT_EMAIL_*`. In MVP without credentials, alerts are logged only.

## 8.10 Runbooks (`docs/runbooks/RB-xx.md`)

Each runbook: symptom → dashboards to check → diagnostic queries (SQL/PromQL) → remediation steps → verification → escalation. Examples: RB-05 Qdrant drift: check sync DAG logs for `AKL-E5020`; run `akl-cli qdrant diff --dry-run`; if Qdrant has extra points, run `akl-cli qdrant sync --delete-orphans`; if Gold has extra, check embedding coverage; verify `akl_qdrant_gold_drift == 0`.

## 8.11 Observability Folder Structure

```
observability/
├── prometheus/
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── recording_rules.yml
├── alertmanager/alertmanager.yml
├── grafana/
│   ├── provisioning/datasources/prometheus.yml
│   ├── provisioning/dashboards/dashboards.yml
│   └── dashboards/*.json
├── otel/otel-collector.yml
└── statsd/statsd_mapping.yml
akl/observability/
├── logging.py        # structlog config, redaction processor, context binding
├── metrics.py        # registry, metric definitions (single source for Appendix F), pushgateway helper
├── tracing.py        # OTel setup, span helpers, decorators
├── lineage.py        # LakehouseIO hooks writing lineage_edges
└── cost.py           # LLM token pricing, compute-hour estimates
```

---

# Chapter 9 — Security & Governance

## 9.1 Threat Model (STRIDE summary)

| Threat | Vector | Mitigation |
|---|---|---|
| Spoofing | Forged JWT / stolen API key | RS256 JWTs with short TTL, key rotation; API keys hashed (argon2) with prefix lookup; IP allow-list optional |
| Tampering | Modifying Bronze/Gold | Content-addressing + versioned bucket + write-only credentials for pipelines; Object Lock optional |
| Repudiation | Unattributed reads/changes | Audit log for every write and every restricted read |
| Information disclosure | Restricted chunk leaked via retrieval | Filter-in-retrieval (ADR-009); reranker and LLM never see unauthorised chunks; response never includes text of chunks outside citations |
| Prompt injection via documents | Malicious instructions inside ingested content | Context blocks are delimited and labelled data; system prompt instructs to treat as data; answer guards check for unsupported tokens; documents from `public` HTML sources are `untrusted=true` payload flag and de-prioritised (×0.9) |
| Denial of service | Query floods, huge uploads | Rate limiting, upload size limits, parser subprocess limits, timeouts |
| Elevation of privilege | Admin endpoints | Separate `admin` scope; service tokens for DAG→API calls with minimal scope |
| Supply chain | Malicious dependency/model | Pinned versions with hashes, model checksums verified (`AKL_EMBED_MODEL_SHA256`) |

## 9.2 Authentication

| Method | Use | Details |
|---|---|---|
| JWT (Bearer) | Interactive users, internal tools | RS256; issuer `AKL_JWT_ISSUER`; JWKS at `AKL_JWT_JWKS_URL` or local key pair in MVP (`AKL_JWT_PRIVATE_KEY_PATH`); claims `sub, email, groups[], security_levels[], scope, exp (≤ 1 h)` |
| API key | Service integrations, CI | Header `X-API-Key: akl_<prefix>_<secret>`; stored `api_keys(prefix, hash, scopes, groups, security_levels, expires_at, last_used_at)` |
| Service token | Airflow → API | JWT with `scope=admin:reload`, minted at container start from local key |

MVP ships a `akl-cli auth mint-token --user dev --groups eng --levels public,internal` for local development. `AKL_AUTH_DISABLED=true` is permitted only when `AKL_ENV=dev`.

## 9.3 Authorization and RBAC

| Role | Scopes | Description |
|---|---|---|
| `reader` | `search:read chat:write` | Query the corpus within their security levels/groups |
| `contributor` | + `documents:write` | Upload documents, trigger GitHub sync for repos they own |
| `curator` | + `quarantine:manage documents:delete` | Manage quarantine, delete documents |
| `admin` | `*` | Reindex, reembed, config, keys |
| `service` | `admin:reload pipelines:trigger` | Machine accounts |

**Security levels**: `public < internal < restricted`. Each document has one level plus optional `allowed_groups` (list). A principal may read a chunk iff `chunk.security_level ∈ principal.security_levels` **and** (`chunk.allowed_groups` is empty **or** `chunk.allowed_groups ∩ principal.groups ≠ ∅`). This predicate is compiled to a Qdrant filter (`must: security_level in [...]; should/must: allowed_groups any-of groups OR allowed_groups is_empty`) and to the BM25 in-memory filter. It is unit-tested exhaustively (Chapter 12).

Assignment: connector config `security_level` + `path_rules`; frontmatter/PDF metadata may **raise** but never lower the level; admin API can override per document (audited).

## 9.4 Secrets Management

- No secrets in repo. `.env.example` documents keys; `.env` git-ignored.
- Compose reads secrets from `secrets/*.txt` files mounted as Docker secrets (`/run/secrets/…`) for Postgres passwords, MinIO keys, JWT private key, GitHub token, LLM API key; env vars named `*_FILE` point to them (`AKL_DB_PASSWORD_FILE`).
- Rotation procedure documented in `docs/security/rotation.md`; API keys rotated via Admin API (`POST /v1/admin/api-keys/{id}/rotate`).
- Enterprise Scale: Vault / cloud KMS; External Secrets Operator on Kubernetes.

## 9.5 Encryption

- **In transit**: Compose network is private; TLS terminated at `nginx`/`traefik` reverse proxy in `docker-compose.prod.yml` with self-signed or ACME certs; MinIO and Qdrant TLS optional in MVP, required in prod profile; Postgres `sslmode=require` in prod.
- **At rest**: MinIO SSE-S3 with KMS (`MINIO_KMS_*`) in prod profile; Postgres volume encryption delegated to host disk encryption; Qdrant snapshots encrypted before upload to backup prefix (`age`/`gpg`) when `AKL_BACKUP_ENCRYPT=true`.

## 9.6 Audit Logging and Access Logs

Table `audit_log(id, ts, principal_id, action, resource_type, resource_id, request_id, ip, user_agent, outcome, details JSONB)`. Audited actions: login/token mint, api-key create/rotate/revoke, document upload/delete/override-security, quarantine actions, reindex/reembed, config change, **every read of a `restricted` chunk** (`action=restricted_read` with chunk ids), admin reload. Access logs: every HTTP request as a structured log line (`event=http_access`) with route, status, latency, principal hash — never the query text at `info` level unless `AKL_LOG_QUERIES=true` (dev only).

## 9.7 Document Permissions Lifecycle

Security level and groups are denormalised into Silver chunks, Gold units, and Qdrant payload. A permission change on a document therefore requires: Postgres `documents` update → Gold `retrieval_units` rewrite for that document's rows → Qdrant `set_payload` for those point ids → BM25 payload refresh (on next rebuild; interim in-memory patch via reload). This is implemented by `akl.governance.permissions.apply(document_id, level, groups)` and exposed as `PATCH /v1/admin/documents/{id}/permissions`; the change takes effect in retrieval within seconds (Qdrant payload update is synchronous).

## 9.8 PII Handling

- Ingestion PII scanner (`presidio-analyzer` optional; default regex set for emails, phone numbers, IBAN, credit cards, national ID patterns) tags documents with `pii_types[]` and `pii_count`.
- Policy: `AKL_PII_POLICY = flag | redact | quarantine` (default `flag`). `redact` replaces matches with typed placeholders (`<EMAIL>`) in Silver text (Bronze remains raw and access-restricted); `quarantine` blocks.
- PII-tagged documents default to `security_level ≥ internal`.

## 9.9 GDPR Considerations

- Right of access: `GET /v1/admin/subjects/search?email=…` returns documents/chunks containing the identifier (via PII index table `pii_mentions(document_id, chunk_id, pii_type, hash)`).
- Right to erasure: hard-delete workflow (9.15) with Bronze object deletion (Object Lock must be off or expired; documented constraint).
- Data minimisation: conversation TTL; query text logging off by default; principal IDs hashed in logs.
- Processing records: audit log and lineage tables serve as the processing register.

## 9.10 Data Retention

| Data | Retention | Mechanism |
|---|---|---|
| Bronze raw | `AKL_BRONZE_RETENTION_DAYS` (365) for versions no longer current | Maintenance DAG + MinIO lifecycle rule as backstop |
| Silver superseded versions | 180 days | Maintenance DAG |
| Gold retired embedding versions | 30 days after cutover | Maintenance DAG |
| Quarantine | 90 days | Maintenance DAG |
| Conversations | 30 days | Maintenance DAG |
| Audit log | 2 years (never auto-deleted in MVP) | Partitioned table by month |
| Backups | 14 days | Maintenance DAG |
| Metrics | Prometheus 30 d local | `--storage.tsdb.retention.time` |

## 9.11 Document Deletion Workflow

```mermaid
sequenceDiagram
    participant A as Curator/Admin or GitHub deletion event
    participant API
    participant PG as Postgres
    participant G as Gold/Silver
    participant Q as Qdrant
    participant B as Bronze
    A->>API: DELETE /v1/documents/{id}?mode=soft|hard
    API->>PG: documents.status='deleting'; audit_log
    API->>PG: mark chunks is_current=false, is_deleted=true
    API->>G: append tombstone rows (silver.documents, silver.chunks); rewrite gold.retrieval_units partition rows for document
    API->>Q: delete points by filter document_id
    API->>API: BM25 in-memory exclusion set += chunk_ids (until next rebuild)
    alt hard
        API->>G: rewrite gold.chunk_embeddings files removing chunk_ids; delete embedding_cache rows for their embedded_text_sha256 if not referenced elsewhere
        API->>B: delete bronze raw object if sha unreferenced; manifest tombstone row
        API->>PG: purge messages.citations referencing lineage (replace with [deleted])
    end
    API->>PG: documents.status='deleted'; audit_log
```

Soft delete is the default and is reversible (`POST /v1/documents/{id}/restore` within retention). Hard delete is irreversible and requires `documents:delete` scope plus `X-Confirm: hard-delete` header.

## 9.12 Governance Folder Structure

```
akl/security/
├── auth/            # jwt.py, api_keys.py, principal.py, dependencies.py (FastAPI Depends)
├── rbac.py          # scopes, roles, security predicate → Qdrant/BM25 filters
├── audit.py
├── secrets.py       # *_FILE resolution
└── ratelimit.py
akl/governance/
├── permissions.py
├── deletion.py
├── pii.py
├── retention.py
└── gdpr.py
```

---

# Chapter 10 — Backend APIs

## 10.1 API Design Rules

- Base path `/v1`. JSON request/response; `application/x-ndjson` or SSE for streams; `multipart/form-data` for uploads.
- Every response includes `X-Request-ID`; errors follow the error model (10.10).
- Pydantic v2 models in `akl/api/schemas/`; OpenAPI generated with tags, examples, and `operationId`s; served at `/docs` and `/openapi.json` (disabled in prod unless `AKL_OPENAPI_ENABLED=true`).
- Pagination: cursor-based (`?cursor=&limit=` ≤ 200) for list endpoints; response `{items, next_cursor, total_estimate}`.
- Versioning: path version; breaking changes → `/v2`.
- Idempotency: `POST /v1/documents` accepts `Idempotency-Key` header (stored 24 h in `idempotency_keys`).
- Rate limiting: token bucket per principal (`AKL_RATE_LIMIT_RPM` default 120; chat 30) with `429` + `Retry-After`; storage in Postgres (MVP) with in-memory fast path; Redis at Enterprise Scale.

## 10.2 Endpoint Inventory

| Method | Path | Scope | Purpose |
|---|---|---|---|
| POST | `/v1/documents` | `documents:write` | Upload PDF/MD/HTML file(s) |
| GET | `/v1/documents` | `search:read` | List documents (filters: source_type, status, security_level, q) |
| GET | `/v1/documents/{document_id}` | `search:read` | Document detail + versions + chunk stats |
| GET | `/v1/documents/{document_id}/chunks` | `search:read` | Paginated chunks (authorised only) |
| DELETE | `/v1/documents/{document_id}` | `documents:delete` | Soft/hard delete |
| POST | `/v1/documents/{document_id}/restore` | `documents:delete` | Restore soft-deleted |
| POST | `/v1/sources/github/sync` | `documents:write` | Trigger sync for a configured repo (`repo`, `branch`, `full=false`) |
| GET | `/v1/sources` | `search:read` | Connector configs + state + health |
| POST | `/v1/search` | `search:read` | Hybrid/dense/sparse search |
| POST | `/v1/chat` | `chat:write` | Answer with citations; `stream` flag |
| GET | `/v1/conversations/{id}` | `chat:write` | Conversation with turns |
| DELETE | `/v1/conversations/{id}` | `chat:write` | Delete own conversation |
| GET | `/v1/health` | none | Liveness |
| GET | `/v1/health/ready` | none | Readiness (models loaded, deps reachable) |
| GET | `/v1/health/dependencies` | none | Per-dependency status |
| GET | `/metrics` | none (network-restricted) | Prometheus |
| GET | `/v1/admin/pipelines` | `admin` | DAG run states (proxy to Airflow) |
| POST | `/v1/admin/pipelines/{dag_id}/trigger` | `admin` or `pipelines:trigger` | Trigger with conf |
| GET | `/v1/admin/quarantine` | `quarantine:manage` | List |
| POST | `/v1/admin/quarantine/{id}/retry` / `dismiss` | `quarantine:manage` | |
| POST | `/v1/admin/reindex` | `admin` | Blue/green Qdrant rebuild |
| POST | `/v1/admin/reembed` | `admin` | New embedding version backfill |
| POST | `/v1/admin/reload-index` | `admin:reload` | Hot-swap BM25/vocab |
| PATCH | `/v1/admin/documents/{id}/permissions` | `admin` | Level/groups |
| GET | `/v1/admin/lineage/document/{id}` | `admin` | Lineage graph |
| GET | `/v1/admin/stats` | `admin` | Corpus stats snapshot |
| GET/POST/DELETE | `/v1/admin/api-keys` | `admin` | Key management |
| GET/PUT | `/v1/admin/config/retrieval` | `admin` | Runtime thresholds (min confidence, k's) |

## 10.3 Upload API

`POST /v1/documents` — multipart: `files[]` (1–20, each ≤ `AKL_MAX_UPLOAD_MB` 50), `security_level` (default `internal`), `allowed_groups[]`, `metadata` (JSON string), `process=async|sync` (sync parses inline for single small file ≤ 5 MB; else async).

Response `202 Accepted` (async): `{items: [{filename, document_id, content_sha256, status: "bronze", dedup: bool}], run_id, status_url}`; `200` (sync) with `status: "silver"` and parse summary. Errors: `413` too large, `415` unsupported, `422` validation, `409` idempotency conflict.

Behaviour: stream to temp file → sha256 → MIME sniff → Bronze write (3.7) → Postgres → trigger `akl_ingestion` with `conf.document_ids` (debounced: at most one trigger per 60 s; documents are picked up by the scheduled run regardless).

## 10.4 GitHub Sync API

`POST /v1/sources/github/sync` body `{repo: "owner/name", branch?: str, full?: bool}` → validates repo exists in `configs/connectors`, checks principal may sync it (`owners` in config), triggers `akl_ingestion` with `conf.connectors=[id], conf.full=full` → `202 {run_id, status_url}`. `GET /v1/sources` returns each connector's `last_commit_sha`, `last_run_at`, `documents_count`, health.

## 10.5 Search API

Request:

```json
{ "query": "how is the embedding cache keyed", "mode": "hybrid|dense|sparse", "k": 10,
  "filters": { "source_type": ["github"], "repo": ["org/akl-docs"], "chunk_type": ["prose","code"], "updated_after": "2026-01-01" },
  "rerank": true, "include_text": true, "precision": "default|high" }
```

Response `200`: `{results: [{chunk_id, lineage_id, document_id, title, source_type, locator, url, heading_breadcrumb, text?, scores: {dense, sparse, rrf, rerank}, rank}], query: {normalized, corrected, intent, entities, filters_applied}, timings_ms, gold_snapshot_id}`. `k ≤ 50`. `text` omitted when `include_text=false` (payload-light).

## 10.6 Chat API

Request: `{query, conversation_id?, stream: false, filters?, k?: 8, mode?: "auto|generative|extractive", include_trace: false}`. Non-stream response = Chapter 6.11 body. Stream = SSE events (6.9). New conversation created when `conversation_id` absent; returned in `meta`. `403` if conversation belongs to another principal.

## 10.7 Health API

- `/v1/health` → `200 {status:"ok", version, git_sha}` always when process alive.
- `/v1/health/ready` → `200` only when: Postgres reachable, Qdrant alias resolves and payload version matches `AKL_EMBEDDING_VERSION`, MinIO bucket accessible, BM25 index loaded (or `AKL_ALLOW_SPARSE_UNAVAILABLE=true`), models loaded. Else `503 {failing: [...]}`.
- `/v1/health/dependencies` → per-dependency latency and status, used by Compose healthcheck and Grafana.

## 10.8 Admin API Semantics

- `reindex`: body `{target_collection?: "kb_chunks_v2", hnsw?: {...}, quantization?: {...}}` → creates job `admin_jobs(job_id, type, status, progress, started_by)`, triggers `akl_qdrant_sync` with `conf.reindex=true` → `202 {job_id}`; `GET /v1/admin/jobs/{job_id}` for progress.
- `reembed`: body `{embedding_version: "...", model_id, model_version}` → validates model available; sets `pending_embedding_version`; triggers `akl_embedding` with `conf.embedding_version`; when coverage = 1.0 and reindex completes, admin promotes via `POST /v1/admin/embedding-version/promote`.
- `config/retrieval`: PUT `{min_confidence, dense_k, sparse_k, fused_k, top_k, rrf_k, rerank_enabled}` → stored in `runtime_config` table; API reloads within 10 s; audited.

## 10.9 Pydantic Model Inventory (`akl/api/schemas/`)

`common.py` (ErrorResponse, Page[T], Timings), `documents.py` (UploadResponse, DocumentSummary, DocumentDetail, ChunkSummary, DeleteRequest), `sources.py`, `search.py` (SearchRequest, SearchFilters, SearchResult, SearchResponse, Scores), `chat.py` (ChatRequest, ChatResponse, Citation, RetrievalInfo, StreamEvent variants), `health.py`, `admin.py` (TriggerRequest, QuarantineItem, ReindexRequest, ReembedRequest, PermissionsPatch, RetrievalConfig, ApiKeyCreate/Response, JobStatus). All models: `model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)`; enums for `source_type`, `security_level`, `chunk_type`, `mode`.

## 10.10 Error Model

```json
{ "error": { "code": "AKL-E6012", "message": "Retrieval backends unavailable", "details": {...}, "request_id": "…", "retryable": true, "docs_url": "https://…/errors#AKL-E6012" } }
```

| HTTP | Usage |
|---|---|
| 400 | Malformed request (non-schema) |
| 401 | Missing/invalid credentials |
| 403 | Insufficient scope / not owner |
| 404 | Resource not found (never reveals existence of unauthorised documents — returns 404 for forbidden document ids) |
| 409 | Conflict (idempotency, state) |
| 413 / 415 / 422 | Upload/validation |
| 429 | Rate limited |
| 500 | Unexpected; `stack_hash` logged |
| 503 | Dependency unavailable / not ready |

Exception hierarchy `akl.errors.AKLError(code, http_status, retryable)` → subclasses per domain; a single exception handler maps to the model. Full catalog: Appendix G.

## 10.11 API Folder Structure

```
akl/api/
├── main.py              # app factory, lifespan (warm models, load index), middleware stack
├── deps.py              # DI: settings, db session, qdrant client, rag service, principal
├── middleware/
│   ├── request_id.py
│   ├── logging.py
│   ├── ratelimit.py
│   └── metrics.py
├── routers/
│   ├── documents.py
│   ├── sources.py
│   ├── search.py
│   ├── chat.py
│   ├── conversations.py
│   ├── health.py
│   └── admin/
│       ├── pipelines.py
│       ├── quarantine.py
│       ├── index.py       # reindex, reembed, reload
│       ├── documents.py   # permissions
│       ├── lineage.py
│       ├── stats.py
│       ├── api_keys.py
│       └── config.py
├── schemas/…
├── streaming.py         # SSE utilities
└── errors.py            # handlers
```

---

# Chapter 11 — Infrastructure

## 11.1 Docker Compose Architecture

Three Compose files layered with `-f`:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Core stack (all 11 services + pushgateway + otel collector) |
| `docker-compose.dev.yml` | Bind-mounts source for hot reload, exposes all ports, `AKL_AUTH_DISABLED=true`, debug logging |
| `docker-compose.prod.yml` | Reverse proxy with TLS, no bind mounts, resource limits, restart policies, secrets from files, MinIO SSE |
| `docker-compose.observability.yml` (profile) | Loki, Promtail, Tempo, cAdvisor — optional |

```mermaid
flowchart TB
    subgraph net_edge["network: akl-edge"]
        PX[traefik/nginx :443]
        API[akl-api :8000]
        AFW[airflow-webserver :8080]
        GRF[grafana :3000]
        MLF[mlflow :5000]
        MC[minio console :9001]
    end
    subgraph net_data["network: akl-data (internal)"]
        PG[(postgres :5432)]
        QD[(qdrant :6333/6334)]
        MIO[(minio :9000)]
        AFS[airflow-scheduler]
        AFT[airflow-triggerer]
        AFWK[airflow-worker]
        PGW[pushgateway :9091]
        PRM[prometheus :9090]
        OTC[otel-collector :4317]
    end
    PX --> API & AFW & GRF & MLF & MC
    API --> PG & QD & MIO & OTC & AFW
    AFS --> PG & MIO & QD & PGW & OTC & MLF
    AFWK --> PG & MIO & QD & PGW
    AFW --> PG
    MLF --> PG & MIO
    PRM --> API & PGW & QD & MIO & PG & AFW
    GRF --> PRM
```

## 11.2 Container Inventory

| Service | Image | Ports (host:container) | Networks | Volumes | Depends on | Healthcheck |
|---|---|---|---|---|---|---|
| `akl-api` | `akl/api:<ver>` (built from `docker/api.Dockerfile`, python:3.12-slim) | 8000:8000 | edge, data | `akl_models:/models`, `akl_tmp:/tmp/akl` | postgres(healthy), qdrant(healthy), minio(healthy) | `GET /v1/health/ready` 20 s |
| `airflow-scheduler` | `akl/airflow:<ver>` (apache/airflow:2.10-python3.12 + akl) | — | data | `./airflow/dags:/opt/airflow/dags`, `akl_airflow_logs`, `akl_models` | postgres, airflow-init(completed) | `airflow jobs check --job-type SchedulerJob` |
| `airflow-webserver` | same | 8080:8080 | edge, data | logs | airflow-init | `GET /health` |
| `airflow-triggerer` | same | — | data | logs | airflow-init | `airflow jobs check --job-type TriggererJob` |
| `airflow-worker` | same | — | data | logs, models | airflow-init | idle in LocalExecutor mode (`profiles: [celery]`) |
| `airflow-init` | same | — | data | — | postgres | one-shot: `db migrate`, create admin user, import pools/connections/variables |
| `postgres` | `postgres:16-alpine` | 5432:5432 (dev only) | data | `akl_pg_data:/var/lib/postgresql/data`, `./docker/postgres/init:/docker-entrypoint-initdb.d` | — | `pg_isready` |
| `qdrant` | `qdrant/qdrant:v1.12` | 6333:6333, 6334:6334 (dev) | data | `akl_qdrant_data:/qdrant/storage`, `akl_qdrant_snapshots:/qdrant/snapshots` | — | `GET /readyz` |
| `minio` | `minio/minio:RELEASE.2025-xx` | 9000:9000, 9001:9001 | edge(console), data | `akl_minio_data:/data` | — | `GET /minio/health/live` |
| `minio-init` | `minio/mc` | — | data | — | minio(healthy) | one-shot: create bucket, enable versioning, lifecycle rules, service users/policies |
| `prometheus` | `prom/prometheus:v2.5x` | 9090:9090 (dev) | data | `akl_prom_data`, `./observability/prometheus:/etc/prometheus:ro` | — | `GET /-/healthy` |
| `pushgateway` | `prom/pushgateway` | — | data | — | — | `GET /-/healthy` |
| `alertmanager` | `prom/alertmanager` | 9093 (dev) | data | config | — | `GET /-/healthy` |
| `grafana` | `grafana/grafana:11` | 3000:3000 | edge, data | `akl_grafana_data`, provisioning ro | prometheus | `GET /api/health` |
| `mlflow` | `akl/mlflow:<ver>` (mlflow + psycopg + boto3) | 5000:5000 | edge, data | — | postgres, minio | `GET /health` |
| `otel-collector` | `otel/opentelemetry-collector-contrib` | 4317 (internal) | data | config ro | — | `GET :13133` |
| `traefik` (prod) | `traefik:v3` | 80, 443 | edge | certs volume | — | ping |

## 11.3 Networking

- `akl-edge`: bridge; only user-facing services attach; in prod only `traefik` publishes host ports.
- `akl-data`: `internal: true` in prod (no egress except through API/scheduler which need GitHub/LLM egress — therefore `akl-data` is not internal in MVP; prod adds an `akl-egress` network attached to api and scheduler only).
- Service discovery by Compose DNS names (`postgres`, `qdrant`, `minio`, …). All AKL config uses these names by default (`AKL_DB_HOST=postgres`).

## 11.4 Volumes

| Volume | Content | Backup? | Size guidance |
|---|---|---|---|
| `akl_pg_data` | Postgres cluster (akl, airflow, mlflow DBs) | yes (pg_dump) | 1–5 GB |
| `akl_minio_data` | Lakehouse bucket, backups prefix, MLflow artefacts | yes (mirror) | dominant; 2–5× raw corpus |
| `akl_qdrant_data` | Collections | rebuildable; snapshots kept | ~2 KB/point |
| `akl_qdrant_snapshots` | Snapshot files | copied to MinIO | |
| `akl_models` | HF/ONNX model files (bge, reranker, spell dict) | no (re-downloadable; checksummed) | ~500 MB |
| `akl_airflow_logs` | Task logs | no | rotate 14 d |
| `akl_prom_data` | TSDB | no | 30 d retention |
| `akl_grafana_data` | Grafana state (dashboards are provisioned from repo, so mostly users/prefs) | no | small |
| `akl_tmp` | Parser scratch | no | tmpfs in prod |

## 11.5 Environment Variables and Secrets

Complete list in Appendix B. Conventions: `AKL_` prefix for application; upstream services use their native names (`POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, `AIRFLOW__CORE__EXECUTOR`, `MLFLOW_S3_ENDPOINT_URL`). Every secret has a `_FILE` variant. `akl.config.Settings` (pydantic-settings) validates at startup and fails fast with a list of missing/invalid keys (`AKL-E0001`).

## 11.6 Local Development Workflow

1. `cp .env.example .env`; edit tokens (GitHub, optional LLM).
2. `make up` → `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build`.
3. `make wait` → polls `/v1/health/ready`, Airflow `/health`, MinIO console.
4. `make seed` → uploads `examples/docs/*.md|pdf` and triggers `akl_ingestion` → `akl_chunking` → `akl_embedding` → `akl_qdrant_sync` (or `make pipeline` to run DAGs sequentially via CLI `airflow dags test`).
5. `make token` → prints a dev JWT. `make query Q="how is bronze keyed"`.
6. `make test-unit`, `make test-integration` (against running stack), `make lint`, `make fmt`.
7. Hot reload: `akl-api` runs `uvicorn --reload` with source bind-mounted in dev; Airflow re-parses DAGs from bind mount.
8. `make down` (keep volumes) / `make nuke` (remove volumes).

CLI `akl-cli` (Typer) mirrors admin operations: `akl-cli ingest run --connector …`, `akl-cli chunk run`, `akl-cli embed run`, `akl-cli qdrant sync|diff|reindex`, `akl-cli eval run`, `akl-cli auth mint-token`, `akl-cli lakehouse compact|stats|query "<sql>"`.

## 11.7 Startup Sequence

```mermaid
sequenceDiagram
    participant PG as postgres
    participant MIO as minio
    participant MI as minio-init
    participant QD as qdrant
    participant AI as airflow-init
    participant AS as airflow-scheduler/webserver/triggerer
    participant ML as mlflow
    participant API as akl-api
    participant PR as prometheus/grafana
    PG->>PG: init scripts create DBs akl, airflow, mlflow + roles
    MIO->>MIO: start
    MI->>MIO: mc alias; mb akl-lakehouse; versioning on; lifecycle; users akl_api (rw), akl_pipeline (rw), akl_mlflow (rw mlflow/*)
    QD->>QD: start (collection created by sync DAG / API ensure step)
    AI->>PG: airflow db migrate; users create; pools import; variables/connections import
    AS->>PG: start after airflow-init exit 0
    ML->>PG: backend store; artifact root s3://akl-lakehouse/mlflow
    API->>PG: alembic upgrade head (akl schema)
    API->>API: load settings, warm models, ensure Qdrant collection+indexes (idempotent), load BM25 if present
    API-->>PR: /metrics available; readiness true
```

`depends_on` with `condition: service_healthy` / `service_completed_successfully` enforces this ordering.

## 11.8 Health Checks

Each service's healthcheck (11.2) uses `interval: 15s, timeout: 5s, retries: 10, start_period: 60s` (API `start_period: 180s` for model warm-up). Compose restart policy `unless-stopped` (prod) / `on-failure` (dev).

## 11.9 Persistent Storage Sizing (MVP reference: 2,000 docs, 100k chunks)

| Store | Estimate |
|---|---|
| Bronze raw | ≈ raw corpus (e.g. 1.5 GB) |
| Silver documents + chunks | ≈ 0.4× raw (ZSTD) |
| Gold retrieval_units | ≈ 0.3× raw |
| Gold embeddings | 100k × 1.5 KB ≈ 150 MB |
| Qdrant | ≈ 250–350 MB |
| Postgres | ≈ 400 MB (cache 150 MB) |
| Total | ≈ 3–4 GB |

## 11.10 Backup Strategy

| Component | Method | Schedule | Destination |
|---|---|---|---|
| Postgres `akl` | `pg_dump -Fc` | daily (maintenance DAG) | `s3://akl-lakehouse/backups/postgres/` |
| Postgres `airflow`, `mlflow` | `pg_dump` | daily | same |
| MinIO bucket | Bucket versioning (point-in-time); `mc mirror` to secondary path/bucket when `AKL_BACKUP_MIRROR_TARGET` set | continuous / daily | secondary |
| Qdrant | Collection snapshot API | daily | `backups/qdrant/` |
| Grafana | dashboards are code | — | git |

## 11.11 Restore Strategy

1. Restore MinIO data (volume restore or `mc mirror` back).
2. `pg_restore` `akl`, `airflow`, `mlflow` databases.
3. Qdrant: either restore snapshot (`POST /collections/kb_chunks_v1/snapshots/upload`) **or** run `akl-cli qdrant sync --full` (rebuild from Gold; preferred — validates ADR-001).
4. Rebuild BM25: `akl-cli bm25 build`.
5. Verify: `akl-cli verify --all` (counts across Postgres, Gold, Qdrant; `akl_qdrant_gold_drift == 0`).
Documented RTO (MVP): < 1 h for 100k chunks; RPO: 24 h (daily) or minutes (with versioning + WAL archiving in prod).

## 11.12 Free Cloud Deployment Strategy

Target: demonstrate a public instance at zero/near-zero cost.

| Option | Approach | Limits |
|---|---|---|
| Single VM (Oracle Cloud Always-Free ARM 4 OCPU/24 GB, or GCP e2-micro + trial) | Same Compose stack; `docker-compose.prod.yml`; Traefik + Let's Encrypt; swap Pushgateway retention | 24 GB fits full stack with 2k docs; ARM images required (`platform: linux/arm64` build) |
| Object storage swap | Replace MinIO with Cloudflare R2 free tier (10 GB) or Backblaze B2 via `AKL_S3_ENDPOINT` | Egress-free for R2 |
| Managed Postgres | Neon/Supabase free tier for `akl` DB only | Airflow/MLflow remain local |
| Qdrant Cloud free (1 GB) | `AKL_QDRANT_URL` + API key | ~300k vectors |
| Airflow | Keep on VM; or replace scheduler with GitHub Actions cron invoking `akl-cli` (documented degraded mode) | |

## 11.13 Migration Path to Kubernetes

| Compose element | Kubernetes equivalent |
|---|---|
| `akl-api` service | Deployment (HPA on CPU + custom metric `akl_http_requests_inflight`), Service, Ingress |
| Airflow containers | Official Airflow Helm chart, `KubernetesExecutor`; each task = pod; `akl_embedding` shards request GPU node pool via `executor_config` |
| Postgres | Managed (RDS/Cloud SQL) or CloudNativePG operator |
| Qdrant | Qdrant Helm chart / Qdrant Cloud, sharded, replicated |
| MinIO | Managed S3/GCS; or MinIO Operator |
| Prometheus/Grafana | kube-prometheus-stack; ServiceMonitors for AKL |
| Secrets | External Secrets Operator |
| Volumes | PVCs; models via init-container download to emptyDir or a read-only PVC |
| Config | ConfigMaps from `configs/`; Helm values mirror `.env` keys |

`deploy/helm/akl/` is reserved in the repo tree (empty in MVP with a README describing values mapping).

---

# Chapter 12 — Testing Strategy

## 12.1 Test Pyramid and Targets

| Layer | Scope | Runner | Target coverage / count |
|---|---|---|---|
| Unit | Pure functions and classes with mocked IO | `pytest -m unit` (no services) | ≥ 85% of `akl/` lines |
| Component | One module against a real dependency (Postgres via `testcontainers`, MinIO, Qdrant) | `pytest -m component` | key repositories/clients |
| Integration / pipeline | DAG task functions end-to-end on the Compose stack with the seed corpus | `pytest -m integration` | 5 DAGs, upload→answer flow |
| API | HTTP contract tests via `httpx.AsyncClient(app)` and against live stack | `pytest -m api` | every endpoint, every error code |
| Evaluation | Retrieval/RAG quality on synthetic + curated set | `pytest -m eval` (nightly) and DAG | thresholds in 0.7 |
| Load | Locust scenarios | manual/nightly | NFR-01/02 |

Overall gate: `≥ 80%` line coverage (NFR-11), `pytest-cov` with `--cov-fail-under=80`.

## 12.2 Unit Tests (representative inventory)

- Parsers: golden files in `tests/fixtures/{pdf,md,html,github}/` with expected `UnifiedDocument` JSON snapshots (`syrupy`); heading detection, table extraction, hyphenation, boilerplate removal, MDX stripping.
- Validators: each rule pass/fail; secret patterns.
- Chunking: property-based tests (`hypothesis`) — no chunk > max tokens; coverage of document text (union of chunk spans == prose spans); determinism (same input → same ids); overlap ≤ configured; code blocks never split below size; table header repetition; merge/split algorithms; incremental diff classification for each event in 4.10 table.
- Identity: `chunk_key/checksum/id` stability across whitespace/heading changes.
- Embedding: provider mock returns fixed vectors; batcher sorting; cache hit/miss; normalisation.
- Fusion/rerank: RRF math; weights; marginal penalty; confidence gate.
- RBAC predicate: exhaustive truth table over levels × groups → Qdrant filter JSON snapshot and in-memory predicate agree.
- Citations: marker parsing/rewriting; invalid markers; extractive fallback triggers.
- Config: settings validation; `_FILE` secret resolution.

## 12.3 Component Tests

- Postgres repositories (documents, versions, chunks, cache, audit) with real schema via Alembic.
- Lakehouse IO: write/read Parquet to MinIO, partition pruning (assert DuckDB `EXPLAIN` shows file skipping), schema enforcement failure.
- Qdrant client: collection ensure idempotency, payload indexes, filtered search, alias swap, reconciler with seeded drift.
- BM25 builder/index: tokenizer behaviours (identifier splitting), filter application, serialisation round-trip.

## 12.4 Pipeline Tests

`tests/integration/test_pipeline_e2e.py`: seed 12 documents (3 per source, including one duplicate, one near-duplicate, one quarantine case, one restricted) → run DAG task functions in order (via `airflow tasks test` or direct invocation with a fake Airflow context) → assertions: Bronze objects/manifest rows; Silver documents/chunks counts; dedup ledger; quarantine item; Gold units exclude low-quality/dup; embeddings coverage 1.0; Qdrant count == Gold; BM25 terms > 0; **second run performs zero writes** (assert metrics counters unchanged and no new Parquet files) — the incremental guarantee test; modify one document → exactly its changed chunks re-embedded; delete one → points removed.

## 12.5 API Tests

Every endpoint: success, auth failure (401/403), validation (422), not found (404 including forbidden-as-404), rate limit (429), streaming event sequence (`meta → token* → citations → done`), idempotency key replay (same response), upload dedup flag, restricted chunk never appears for `internal`-only principal (leakage test over full corpus).

## 12.6 Embedding Tests

Real model test (marked `slow`): embed 3 known sentences, assert dim 384, unit norm ± 1e-4, cosine(sim pair) > cosine(dissim pair); ONNX vs PyTorch parity within 1e-3; INT8 parity within 2e-2; throughput smoke ≥ 100 chunks/s on CI runner.

## 12.7 Retrieval and RAG Evaluation

**Datasets**:
- `gold/eval/qa_pairs` version `v1` generated by `akl-cli eval generate`: for each of N sampled chunks (stratified by source and chunk_type), produce 1–2 questions via (a) heading-to-question templates and (b) LLM generation when available; `expected_chunk_ids` = source chunk (+ neighbours if overlap). Also 50 hand-curated questions in `tests/eval/curated_v1.jsonl` including identifier lookups, code queries, and 20 **unanswerable** questions for refusal calibration.

**Metrics** (computed by `akl.eval`): Recall@k (k=1,5,10), MRR, nDCG@10, hit rate after rerank, mean confidence for answerable vs unanswerable (separation → threshold calibration via ROC, target FPR ≤ 0.1), latency p50/p95, and for RAG (LLM available): citation coverage, faithfulness (NLI or LLM judge), answer relevance, refusal precision/recall. Ablations logged as separate MLflow runs: dense-only, sparse-only, hybrid-no-rerank, hybrid-rerank.

**Thresholds** (fail CI nightly if breached): Recall@10 ≥ 0.85, MRR ≥ 0.70, refusal recall ≥ 0.8 at FPR ≤ 0.1.

## 12.8 Synthetic Dataset Generation

`akl.eval.synth`: sampling strategy, templates (`"What does the section '{heading}' say about {noun phrase}?"`, `"Which {env var|error code} controls {…}?"`), LLM prompt `prompts/eval_qgen_v1.md` with JSON output, deduplication of questions (SimHash), difficulty labelling (single-chunk = easy; requires neighbour = medium; cross-document = hard). Output versioned under `gold/eval/qa_pairs/version=`.

## 12.9 Load Testing

Locust (`tests/load/locustfile.py`): users 20/50/100; mix 70% search, 25% chat (non-stream), 5% upload; ramp 1 min, hold 5 min; assert p95 within NFR-01 at 20 users on laptop; record throughput ceiling. Results appended to `docs/benchmarks/`.

## 12.10 Benchmark Methodology

Fixed corpus snapshot (`gold_snapshot_id` recorded), fixed hardware descriptor (cores, RAM, CPU model captured by `akl-cli bench env`), warm-up 50 queries, 500 measured queries, report medians and p95 with CI; embedding throughput measured over 10k chunks; ingestion throughput over seed set × 10. Every benchmark row: date, git sha, config hash, numbers.

## 12.11 Test Folder Structure

```
tests/
├── conftest.py               # fixtures: settings, tmp MinIO (testcontainers), pg, qdrant, fake embedder
├── unit/
│   ├── ingestion/  chunking/  embedding/  rag/  security/  lakehouse/  api/
├── component/
│   ├── test_pg_repositories.py  test_lakehouse_io.py  test_qdrant_client.py  test_bm25.py
├── integration/
│   ├── test_pipeline_e2e.py  test_incremental_noop.py  test_deletion_cascade.py  test_reindex_bluegreen.py
├── api/
│   ├── test_documents.py  test_search.py  test_chat.py  test_admin.py  test_health.py  test_leakage.py
├── eval/
│   ├── curated_v1.jsonl  test_retrieval_thresholds.py
├── load/locustfile.py
└── fixtures/
    ├── pdf/  md/  html/  github/  golden/
```

---

# Chapter 13 — CI/CD

## 13.1 Branch Strategy

- `main`: protected; always deployable; squash merges only; required checks: `lint`, `unit`, `component`, `api`, `docker-build`.
- `develop`: not used (trunk-based). Feature branches `feat/<area>-<short>`; fix branches `fix/…`; release branches not used — tags cut from `main`.
- Conventional Commits enforced by `commitlint` in CI (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`, `ci:`; scopes = top-level packages).

## 13.2 GitHub Actions Workflows (`.github/workflows/`)

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | PR, push to main | `lint` (ruff check, ruff format --check, mypy strict on `akl/`, yamllint, hadolint, sqlfluff on `sql/`) → `unit` (matrix py3.12; coverage upload) → `component` (services: postgres, minio, qdrant via `services:`) → `api` → `docker-build` (buildx, cache, no push) → `dag-integrity` (`airflow dags list` + `pytest tests/airflow/test_dag_integrity.py`: no import errors, no cycles, all tasks have timeouts) |
| `integration.yml` | nightly cron + manual + label `run-integration` | Spins up full Compose stack on runner, runs seed + `pytest -m integration`, uploads logs artefact |
| `eval.yml` | nightly | Runs retrieval eval against seed corpus, logs to MLflow (file backend in CI), fails on threshold breach, posts summary comment |
| `release.yml` | tag `v*.*.*` | Build multi-arch images (amd64/arm64) → push to GHCR `ghcr.io/<org>/akl-api`, `akl-airflow`, `akl-mlflow` with tags `vX.Y.Z`, `vX.Y`, `latest` → generate SBOM (syft) → sign (cosign keyless) → create GitHub Release with changelog (git-cliff) and `PRD.md`, `openapi.json`, dashboards archive as assets |
| `deploy.yml` | `workflow_dispatch` (env: staging/prod) + on release for staging | SSH to VM (or `docker context`), `docker compose pull && up -d`, run `akl-cli verify --all`, smoke tests, rollback on failure (previous tag) |
| `security.yml` | weekly + PR | `pip-audit`, `trivy image`, `gitleaks`, dependabot config |
| `docs.yml` | push to main | Build MkDocs site (PRD, runbooks, ADRs, API reference from OpenAPI) → GitHub Pages |

## 13.3 Linting and Formatting

- `ruff` (rules: E,F,I,B,UP,N,S,ASYNC,PT) + `ruff format`; `mypy --strict` for `akl/` (Airflow DAG files excluded from strict); `pre-commit` hooks mirror CI.
- SQL: `sqlfluff` dialect duckdb for `sql/`; `sqlfluff` dialect postgres for `migrations/`.
- YAML: `yamllint`; Dockerfiles: `hadolint`; Markdown: `markdownlint` (docs only).

## 13.4 Docker Build

Multi-stage Dockerfiles: `builder` (install with `uv` into venv, compile ONNX exports) → `runtime` (python:3.12-slim, non-root `akl` user, `tini`). Build args `AKL_VERSION`, `GIT_SHA`. Labels per OCI spec. Layer cache via `actions/cache` + buildx `type=gha`. Image size budget: api ≤ 1.2 GB (models excluded, downloaded to volume), airflow ≤ 1.8 GB.

## 13.5 Versioning and Release

- Semantic versioning `MAJOR.MINOR.PATCH`; `MAJOR` for schema-breaking Lakehouse or API changes, `MINOR` for features, `PATCH` for fixes. Version single-sourced in `pyproject.toml`, exposed at `/v1/health`.
- `git-cliff` generates `CHANGELOG.md` from Conventional Commits. Release checklist in `docs/release.md`: bump version → tag → `release.yml` → deploy staging → eval gate → promote prod.
- Data artefact versions independent: `parser_version`, `chunker_version`, `embedding_version`, `prompt_version`, schema versions — each recorded in data and in the release notes' "Data compatibility" section; bumping `chunker_version` in a release triggers automatic re-chunk (documented cost).

## 13.6 Artefacts

Per CI run: coverage XML/HTML, pytest JUnit, eval report (Markdown + Parquet), OpenAPI JSON, DAG graph PNGs (`airflow dags show`), Docker SBOMs. Per release: images, SBOM, signatures, changelog, docs site.

## 13.7 Deployment Pipeline

```mermaid
flowchart LR
    PR[PR opened] --> CI[ci.yml: lint→unit→component→api→build→dag-integrity]
    CI -->|merge| M[main]
    M --> N[nightly integration.yml + eval.yml]
    M -->|tag vX.Y.Z| R[release.yml: multi-arch build, SBOM, sign, GH release]
    R --> DS[deploy.yml staging]
    DS --> V[akl-cli verify + smoke + eval gate]
    V -->|manual approve| DP[deploy.yml prod]
    V -->|fail| RB[rollback previous tag]
```

---

# Chapter 14 — PB Scale Evolution

## 14.1 Invariants That Do Not Change With Scale

1. Layer semantics (Bronze immutable, Silver typed, Gold AI-ready).
2. Identity scheme (content sha, `document_id`, `chunk_key`, `chunk_checksum`, `chunk_id`, `embedding_version`).
3. SQL transformation contracts (`sql/silver/*.sql`, `sql/gold/*.sql`).
4. Vector store as derived state with reconciliation.
5. Retrieval contract (hybrid → fusion → rerank → context → cite).
6. Metric names, error codes, API schemas.

What changes: the engine executing SQL, the table format, the number of Qdrant nodes, the executor running DAG tasks, and the cache backend.

## 14.2 Scale Ladder

| Scale (raw corpus) | Docs / chunks (approx.) | Storage | Compute (transform) | Partitioning | Airflow | Vector | Cache | Notes |
|---|---|---|---|---|---|---|---|---|
| **10 MB** | 50 / 3k | MinIO single disk | DuckDB in-process | Defaults; single file per dataset OK | LocalExecutor, 1 shard | Qdrant single node, no quantisation | Postgres | Laptop demo |
| **1 GB** | 2k–5k / 100k–300k | MinIO single disk (≈4 GB total) | DuckDB, 4 cores | Date + source partitions; compaction daily | LocalExecutor, 4 shards | Single node, ~1 GB RAM | Postgres (≈450 MB) | MVP reference |
| **100 GB** | 200k / 20M | MinIO distributed (4 nodes, erasure coding) or cloud S3 | DuckDB per-partition tasks (each task ≤ 20 GB); begin Spark for cross-partition dedup | Add `repo`/`host` sub-partition for GitHub/HTML; Iceberg adoption **recommended** here | CeleryExecutor, 8–16 workers; pools per source | Qdrant 3-node cluster, 2 shards × 2 replicas, scalar int8 quantisation (30 GB → 8 GB RAM) | Cache index in Postgres, payload in Gold Parquet only | Embedding backlog ≈ 20M × 5 ms = 28 CPU-hours → GPU (T4: ~1 h) |
| **1 TB** | 2M / 200M | Cloud object store, lifecycle tiering (Bronze → infrequent access after 90 d) | **Spark** (or Trino) for Silver/Gold SQL; DuckDB retained for ad-hoc and small partitions | Iceberg tables with hidden partitioning (`days(ingest_date)`, `bucket(64, document_id)`) | KubernetesExecutor; embedding tasks on GPU node pool; DAG fan-out by partition | Qdrant 6–12 nodes, 8 shards, product quantisation for cold tiers; multitenancy by `source_type` collections | Redis/RocksDB keyed cache; Bloom filter of known checksums in memory | BM25 moves to Qdrant sparse vectors (SPLADE or BM25 weights) — single engine hybrid |
| **100 TB** | 200M / 20B | Multi-region replication; Bronze on Glacier-class after 1 y | Spark cluster with autoscaling; incremental Iceberg MERGE; Z-order on `document_id` | Iceberg + partition evolution; metadata tables for pruning | Multiple Airflow deployments per domain or Airflow + Datasets across teams | Vector index tiered: hot (recent/high-traffic) HNSW in RAM, warm on-disk HNSW, cold PQ; routing layer by metadata; ~20B × 100 B (PQ) ≈ 2 TB RAM cluster | Distributed KV (e.g. FoundationDB/DynamoDB) | Reranking becomes the main compute; GPU rerank service with request batching |
| **1 PB** | 2B / 200B | Data mesh: per-domain buckets/catalogs; unified Iceberg REST catalog | Spark/Flink streaming for ingestion (CDC from sources); batch for compaction | Domain-partitioned tables; time-travel for audit | Event-driven orchestration (Airflow for control plane, streaming for data) | Federated vector search: query router → per-domain Qdrant clusters → global fusion; disk-based ANN (DiskANN-class) for cold | Global dedup service | Cost dominated by storage + embedding of change volume, not corpus size (incremental invariant) |

## 14.3 Storage Evolution

- **Object storage**: MinIO single → MinIO distributed (erasure coding, 4–16 drives) → cloud S3/GCS with tiering. Because all IO goes through the S3 API, only `AKL_S3_ENDPOINT` and credentials change.
- **Bronze**: never partitioned finer than `source_type` + content hash; at PB scale add hash-prefix sharding (`sha256=ab/…`) to avoid hot prefixes.
- **Cost optimisation**: ZSTD level 3 (write speed) vs 9 (archive); Bronze lifecycle to cold tier; drop `text` column from Gold `chunk_embeddings` (already absent) and keep text only in `retrieval_units`; Qdrant `on_disk_payload`.

## 14.4 Iceberg Migration (recommended at ≥ 100 GB)

| Concern | Parquet-on-prefix (MVP) | Iceberg |
|---|---|---|
| Atomic multi-file commit | No (manifest table workaround) | Yes (snapshot commit) |
| Schema evolution | Additive by convention | Full (rename, reorder, type promotion) with IDs |
| Partition evolution | Rewrite | Hidden partitioning; evolve without rewrite |
| Time travel / rollback | Manual | `VERSION AS OF` |
| Small-file compaction | Custom DAG task | `rewrite_data_files` procedure |
| Engine support | DuckDB, Spark | DuckDB (read via `iceberg` extension), Spark, Trino, Flink |

Migration steps: stand up Iceberg REST catalog (Nessie/Polaris) → `CREATE TABLE … USING iceberg` for each dataset with same schema → one-time `INSERT INTO … SELECT * FROM read_parquet(…)` → switch `LakehouseIO` writer to Iceberg (Spark/pyiceberg) → readers use catalog → retire manifest workaround. The SQL contracts are unchanged.

## 14.5 Delta Lake Comparison

| | Iceberg | Delta Lake |
|---|---|---|
| Governance | Apache, multi-vendor | Linux Foundation, Databricks-led |
| Partition evolution | Yes | Limited (liquid clustering) |
| Engine breadth | Spark, Trino, Flink, DuckDB, Snowflake, BigQuery | Spark first-class; others via UniForm/Delta Kernel |
| Metadata | Manifest lists; good for very large tables | Transaction log JSON + checkpoints |
| Fit for AKL | Chosen: vendor neutrality matches "open formats" principle | Viable if the organisation is Databricks-centric |

## 14.6 Compute Evolution — Spark Migration

- Trigger: any single transformation regularly exceeds ~20 GB input per task or runs > 30 min on DuckDB.
- Approach: `akl/lakehouse/engine.py` defines `QueryEngine` ABC (`execute_sql`, `write_partitioned`); implementations `DuckDBEngine` (MVP) and `SparkEngine`. SQL files are shared; dialect differences isolated in a small Jinja macro layer (`{{ list_agg }}`, `{{ hash_bucket }}`).
- Airflow tasks switch operator: `@task` (DuckDB) → `SparkSubmitOperator`/`SparkKubernetesOperator`; task ids and XCom contracts unchanged.

## 14.7 Airflow Evolution

LocalExecutor → CeleryExecutor (Redis broker, 8–32 workers, per-queue routing: `default`, `embedding-gpu`, `heavy-parse`) → KubernetesExecutor (pod per task; GPU via `executor_config`; autoscaling). Dynamic task mapping fan-out grows from 4 shards to hundreds keyed by partition. Add `akl_backfill` on-demand DAG parameterised by date range and source. Consider Dataset-triggered "micro-DAGs" per source to isolate blast radius.

## 14.8 Vector Sharding and Distributed Qdrant

- Shard key: `source_type` at 100 GB (collections per source; router fans out and fuses), then `hash(document_id)` shards within a collection at 1 TB+.
- Replication factor 2 for availability; `write_consistency_factor=1` for ingestion speed, reads `consistency=majority` for stale-read tolerance.
- Quantisation: scalar int8 (4×, ~1% recall loss) default at 100 GB; product quantisation (16–32×) for cold collections with rescoring (`rescore=true`) from original vectors on disk.
- Reconciliation becomes partition-scoped: `qdrant_sync_ops` populated from a Gold **change log** table (`gold/changelog` Iceberg table with `op, chunk_id, embedding_version, ts`) rather than full scroll; a weekly full audit remains.
- Aliases per shard family for blue/green.

## 14.9 Distributed Object Storage

MinIO distributed mode (erasure code EC:4 across 8 drives tolerates 4 failures) → cloud provider durability (11 nines). Cross-region replication for Bronze (source of truth). Bucket per layer for IAM separation; pipeline roles get write-only on Bronze/Silver/Gold and read on all.

## 14.10 Caching Strategy at Scale

| Cache | MVP | Scale |
|---|---|---|
| Embedding cache | Postgres BYTEA | Redis (hot) + Gold Parquet (cold) + Bloom filter for negative lookups |
| Query embedding | in-process LRU | Redis shared across API replicas |
| Result cache | optional in-process | Redis with `gold_snapshot_id` in key; invalidated on sync |
| Reranker outputs | none | Redis keyed `(query_hash, chunk_id)` for hot queries |
| BM25 index | in-process | Qdrant sparse vectors (no in-process index) |

## 14.11 Compaction at Scale

Iceberg `rewrite_data_files` (bin-pack 512 MB), `expire_snapshots` (7 d), `remove_orphan_files`; run per partition with Spark; scheduled by `akl_maintenance` with concurrency by partition. Z-ordering on `(document_id, chunk_index)` for `retrieval_units` improves filtered scans for permission rewrites.

## 14.12 Cost Model (indicative, cloud list prices, per month)

| Scale | Storage | Compute (pipelines) | Vector cluster | Embedding (change volume 2%/month, GPU) | Total order |
|---|---|---|---|---|---|
| 1 GB | ~$0 (local) | $0 | $0 | $0 | $0 |
| 100 GB | ~$10 | ~$150 (spot Spark 20 h) | ~$300 (3 × 8 GB nodes) | ~$20 | ~$500 |
| 1 TB | ~$100 | ~$1k | ~$2.5k | ~$150 | ~$4k |
| 100 TB | ~$5k (tiered) | ~$15k | ~$40k | ~$5k | ~$65k |
| 1 PB | ~$30k (tiered) | ~$80k | ~$250k | ~$30k | ~$400k |

The incremental invariant is the primary cost control: embedding and indexing cost scale with **change volume**, not corpus size.

## 14.13 Scale Evolution Diagram

```mermaid
flowchart LR
    A[Laptop<br/>DuckDB + MinIO + single Qdrant<br/>LocalExecutor] --> B[Small cluster 100 GB<br/>MinIO distributed / S3<br/>Iceberg, DuckDB per partition<br/>Celery, Qdrant 3 nodes int8]
    B --> C[1 TB<br/>Spark SQL same contracts<br/>K8s executor + GPU embed<br/>Qdrant sharded + sparse vectors]
    C --> D[100 TB<br/>Tiered vectors hot/warm/cold<br/>Streaming ingestion<br/>Distributed caches]
    D --> E[1 PB<br/>Data mesh domains<br/>Federated vector search<br/>Event-driven orchestration]
```

---

# Chapter 15 — Repository Blueprint

## 15.1 Complete Repository Tree

```
ai-knowledge-lakehouse/
├── README.md                          # short: what, quickstart (make up/seed/query), links to docs
├── PRD.md                             # this document
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile                           # up, down, nuke, wait, seed, pipeline, test-*, lint, fmt, token, query, bench
├── pyproject.toml                     # project metadata, deps (uv), ruff/mypy/pytest config, entry points (akl-cli, akl.connectors, akl.parsers)
├── uv.lock
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/{ci,integration,eval,release,deploy,security,docs}.yml
│   ├── dependabot.yml
│   ├── CODEOWNERS
│   └── PULL_REQUEST_TEMPLATE.md
├── docker/
│   ├── api.Dockerfile
│   ├── airflow.Dockerfile
│   ├── mlflow.Dockerfile
│   ├── postgres/init/
│   │   ├── 01_databases.sql           # create akl, airflow, mlflow DBs + roles
│   │   └── 02_extensions.sql          # pgcrypto, uuid-ossp
│   ├── minio/init.sh                  # bucket, versioning, lifecycle, users, policies
│   └── traefik/{traefik.yml,dynamic.yml}
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── docker-compose.observability.yml
├── secrets/                           # git-ignored; README.md only committed
├── configs/
│   ├── settings.yaml                  # non-secret defaults consumed by akl.config
│   ├── chunking.yaml
│   ├── embedding.yaml
│   ├── retrieval.yaml                 # k's, rrf_k, thresholds (defaults; runtime_config overrides)
│   ├── intents.yaml
│   ├── security.yaml                  # roles→scopes, level ordering
│   ├── prompts/{answer_v1.md,summarize_v1.md,rewrite_v1.md,eval_qgen_v1.md}
│   ├── connectors/*.yaml
│   └── dags/*.yaml
├── akl/                               # Python package
│   ├── __init__.py                    # __version__
│   ├── config.py                      # Settings (pydantic-settings), _FILE secrets, YAML merge
│   ├── errors.py                      # AKLError hierarchy + catalog loader
│   ├── ids.py                         # namespaces, uuid5 helpers
│   ├── cli/                           # Typer app: ingest, chunk, embed, qdrant, bm25, eval, auth, lakehouse, verify, bench
│   ├── lakehouse/
│   │   ├── io.py                      # LakehouseIO: read/write Parquet on S3, schema enforcement, lineage hooks
│   │   ├── engine.py                  # QueryEngine ABC, DuckDBEngine (httpfs config, union_by_name)
│   │   ├── schemas/{bronze,silver,gold}.py   # pyarrow schemas + versions
│   │   ├── views.py                   # registers v_current_* views
│   │   ├── compaction.py
│   │   ├── stats.py
│   │   └── sql/
│   │       ├── views/{v_current_documents,v_current_chunks,v_gold_active_units,v_embedding_coverage}.sql
│   │       ├── silver/{dedup_candidates}.sql
│   │       └── gold/{retrieval_units,stats_snapshot,embedding_backlog}.sql
│   ├── db/
│   │   ├── session.py                 # SQLAlchemy 2 async/sync engines
│   │   ├── models.py                  # ORM for Appendix A tables
│   │   ├── repositories/{documents,chunks,cache,runs,quarantine,audit,conversations,api_keys,config,lineage}.py
│   │   └── migrations/ (alembic)      # env.py, versions/0001_initial.py ... 
│   ├── ingestion/  (Chapter 3.10)
│   ├── chunking/   (Chapter 4.16)
│   ├── embedding/  (Chapter 5.16)
│   ├── rag/        (Chapter 6.15)
│   ├── security/   (Chapter 9.12)
│   ├── governance/ (Chapter 9.12)
│   ├── observability/ (Chapter 8.11)
│   ├── eval/
│   │   ├── datasets.py  synth.py  metrics.py  runner.py  judge.py  calibrate.py
│   ├── pipelines/                     # service-layer entrypoints called by DAG tasks and CLI (no Airflow imports)
│   │   ├── ingestion.py  chunking.py  embedding.py  qdrant_sync.py  maintenance.py
│   └── api/        (Chapter 10.11)
├── airflow/        (Chapter 7.9)
├── sql/                               # symlink/alias to akl/lakehouse/sql for sqlfluff (or lint path config)
├── observability/  (Chapter 8.11)
├── deploy/
│   ├── helm/akl/README.md             # reserved
│   └── vm/{bootstrap.sh,README.md}    # free-cloud VM setup
├── examples/
│   ├── docs/                          # seed corpus: markdown, pdf, html samples
│   └── notebooks/                     # DuckDB exploration, eval analysis (nbstripout)
├── scripts/
│   ├── seed.sh  wait_for_stack.sh  mint_token.sh  export_openapi.py  gen_dag_graphs.py
├── tests/          (Chapter 12.11)
└── docs/
    ├── index.md  architecture.md  adr/ADR-001..010.md  runbooks/RB-01..10.md
    ├── api.md (generated)  errors.md (generated from catalog)  metrics.md (generated)
    ├── security/rotation.md  release.md  benchmarks/
    └── mkdocs.yml
```

## 15.2 Module Responsibilities (summary of non-obvious modules)

| Module | Responsibility | Must not |
|---|---|---|
| `akl.config` | Single settings object; env > yaml > defaults; `_FILE` resolution; validation | Read env anywhere else |
| `akl.ids` | Deterministic ID derivation (uuid5 namespaces `AKL_NS_DOC`, `AKL_NS_CHUNK`, `AKL_NS_QA`) | Random IDs for domain entities |
| `akl.lakehouse.io` | Only path through which Parquet is read/written; enforces schema, ZSTD, sort, partition layout, lineage | Bypass by raw `boto3` writes in services |
| `akl.pipelines.*` | Framework-agnostic units of work returning result dataclasses | Import Airflow |
| `airflow/dags/*` | Thin: schedule, dependencies, mapping, gates, XCom | Business logic |
| `akl.db.repositories` | All SQL against Postgres | Business decisions |
| `akl.rag.service` | Orchestrates stages; owns trace | Direct client construction (use deps) |
| `akl.observability.metrics` | Every metric object defined here; Appendix F generated from it | Ad-hoc metric creation elsewhere |
| `akl.errors` | Every `AKL-E/W` code; Appendix G generated from it | String error codes elsewhere |

## 15.3 Configuration Files

| File | Schema (pydantic model) | Notes |
|---|---|---|
| `configs/settings.yaml` | `Settings` | Non-secret defaults; env overrides |
| `configs/chunking.yaml` | `ChunkConfig` | Hash → `chunk_config_hash` |
| `configs/embedding.yaml` | `EmbeddingConfig` | model_id, version, batch, device, int8, threads |
| `configs/retrieval.yaml` | `RetrievalConfig` | Defaults; `runtime_config` table overrides |
| `configs/intents.yaml` | `IntentConfig` | Labels, rules, training examples |
| `configs/security.yaml` | `SecurityConfig` | Roles, scopes, level order |
| `configs/connectors/*.yaml` | `ConnectorConfig` subclasses | One file per connector instance; `id` field |
| `configs/dags/*.yaml` | `DagConfig` | Schedules, shards, gates, timeouts |
| `configs/prompts/*.md` | — | Versioned prompt templates |

## 15.4 SQL Migrations (Alembic)

`0001_initial` (all Appendix A tables), `0002_indexes`, `0003_partitions_audit_log` (monthly partitions), further migrations additive. Every migration reversible. CI runs `alembic upgrade head && alembic downgrade base && alembic upgrade head`.

## 15.5 Folder Relationship Diagram

```mermaid
flowchart TB
    CFG[configs/] --> AKL[akl/ package]
    AKL --> API[akl/api]
    AKL --> PIPE[akl/pipelines]
    PIPE --> DAGS[airflow/dags]
    AKL --> CLI[akl/cli]
    AKL --> OBS[akl/observability]
    OBS --> OBSCFG[observability/ dashboards, alerts]
    AKL --> TESTS[tests/]
    DOCKER[docker/ + compose files] --> API & DAGS
    DOCS[docs/] -.generated from.-> AKL
```

---

# Chapter 16 — Implementation Roadmap

Milestones are ordered for sequential implementation. Each lists Objective, Files, Dependencies (milestone IDs), Expected output, and a Validation checklist. Phases: **P0 Foundation → P1 Lakehouse → P2 Ingestion → P3 Chunking → P4 Embedding & Vector → P5 RAG → P6 API → P7 Airflow → P8 Observability → P9 Security/Governance → P10 Testing/Eval → P11 CI/CD & Release → P12 Hardening.**

## P0 — Foundation

**M001 Repository scaffold** — Objective: create tree from Chapter 15 with placeholder READMEs. Files: `README.md, LICENSE, .gitignore, .editorconfig, Makefile (targets stubbed), pyproject.toml (name akl, py3.12), uv.lock`. Deps: —. Output: `uv sync` succeeds; `python -c "import akl"`. Validation: [ ] tree matches 15.1 top level; [ ] `make help` lists targets.

**M002 Settings and secrets** — Objective: `akl.config.Settings` with env > yaml > defaults and `_FILE` resolution. Files: `akl/config.py, configs/settings.yaml, .env.example`. Deps: M001. Output: `Settings()` loads; missing required → `AKL-E0001`. Validation: [ ] unit tests for precedence; [ ] `_FILE` read; [ ] `AKL_ENV` enum.

**M003 Error catalog and IDs** — Objective: `AKLError` hierarchy, catalog loader, uuid5 namespaces. Files: `akl/errors.py, akl/ids.py, configs/errors.yaml`. Deps: M002. Output: every code in Appendix G defined. Validation: [ ] catalog → `docs/errors.md` generator; [ ] `document_id` deterministic test.

**M004 Structured logging** — Objective: structlog JSON with schema 8.2.1, redaction, contextvars. Files: `akl/observability/logging.py`. Deps: M002. Output: JSON lines with `correlation_id`. Validation: [ ] schema keys present; [ ] secret redaction test.

**M005 Metrics registry** — Objective: define all Appendix F metrics; pushgateway helper. Files: `akl/observability/metrics.py`. Deps: M004. Output: `/metrics` text on demand. Validation: [ ] every metric has help/type/labels; [ ] generator for `docs/metrics.md`.

**M006 Tracing setup** — Objective: OTel provider, span decorators, log correlation. Files: `akl/observability/tracing.py`. Deps: M004. Validation: [ ] spans exported to console exporter in test; [ ] `trace_id` in logs.

**M007 Docker Compose core** — Objective: compose files with postgres, minio(+init), qdrant, prometheus, pushgateway, grafana, mlflow, otel-collector; volumes/networks per 11.2. Files: `docker-compose*.yml, docker/postgres/init/*.sql, docker/minio/init.sh, docker/mlflow.Dockerfile`. Deps: M001. Output: `docker compose up` healthy. Validation: [ ] all healthchecks green; [ ] bucket `akl-lakehouse` exists with versioning; [ ] DBs `akl/airflow/mlflow` exist.

**M008 Postgres schema and migrations** — Objective: Alembic + ORM for Appendix A. Files: `akl/db/session.py, models.py, migrations/*`. Deps: M007. Validation: [ ] upgrade/downgrade round-trip; [ ] indexes present.

**M009 Repositories layer** — Objective: repositories for documents, versions, chunks, cache, runs, quarantine, audit, conversations, api_keys, runtime_config, lineage. Files: `akl/db/repositories/*.py`. Deps: M008. Validation: [ ] component tests with testcontainers.

**M010 CLI skeleton** — Objective: Typer app with command groups. Files: `akl/cli/__init__.py, main.py`. Deps: M002. Validation: [ ] `akl-cli --help`.

## P1 — Lakehouse

**M011 Pyarrow schemas** — Objective: schemas for all Bronze/Silver/Gold datasets with versions. Files: `akl/lakehouse/schemas/{bronze,silver,gold}.py`. Deps: M003. Validation: [ ] schema JSON snapshot tests.

**M012 DuckDB engine** — Objective: `QueryEngine` ABC + `DuckDBEngine` (httpfs, S3 creds, `union_by_name`, hive partitioning). Files: `akl/lakehouse/engine.py`. Deps: M007. Validation: [ ] query MinIO parquet in component test.

**M013 LakehouseIO** — Objective: write partitioned Parquet (ZSTD, sort key, kv metadata `akl.schema_version`), read with pruning, schema enforcement (`AKL-E2101`), lineage hook. Files: `akl/lakehouse/io.py`. Deps: M011, M012. Validation: [ ] round-trip; [ ] `EXPLAIN` shows pruning; [ ] bad schema rejected.

**M014 Views** — Objective: SQL views 2.9 registered on engine. Files: `akl/lakehouse/sql/views/*.sql, akl/lakehouse/views.py`. Deps: M013. Validation: [ ] views resolve on empty and seeded data.

**M015 Compaction** — Objective: algorithm 2.8. Files: `akl/lakehouse/compaction.py`. Deps: M013. Validation: [ ] small files merged; [ ] row counts equal; [ ] metrics emitted.

**M016 Corpus stats** — Objective: `gold/stats` snapshot SQL + writer. Files: `akl/lakehouse/stats.py, sql/gold/stats_snapshot.sql`. Deps: M014. Validation: [ ] snapshot columns per 2.6.4.

**M017 Lakehouse CLI** — Objective: `akl-cli lakehouse query|compact|stats`. Files: `akl/cli/lakehouse.py`. Deps: M015, M016.

## P2 — Ingestion

**M018 Ingestion models** — Objective: `SourceItem, FetchedObject, UnifiedDocument, Block*`, `QualityReport`. Files: `akl/ingestion/models.py`. Deps: M003. Validation: [ ] frozen models; [ ] JSON round-trip.

**M019 Connector/Parser registries + base classes** — Files: `akl/ingestion/registry.py, connectors/base.py, parsers/base.py`, entry points in `pyproject.toml`. Deps: M018. Validation: [ ] registry discovers test plugin.

**M020 Connector state persistence** — Files: `akl/ingestion/state.py`, repository `connector_state`. Deps: M009, M019.

**M021 Fingerprinting and language** — Objective: sha256, SimHash, language detection. Files: `akl/ingestion/fingerprint.py, language.py`. Deps: M018. Validation: [ ] SimHash Hamming for near-dup fixtures ≤ 3.

**M022 Bronze writer** — Objective: flow 3.7. Files: `akl/ingestion/bronze_writer.py`. Deps: M013, M020. Validation: [ ] dedup skip; [ ] manifest rows; [ ] Postgres transitions; [ ] crash-ordering test.

**M023 Quarantine writer** — Files: `akl/ingestion/quarantine.py`. Deps: M022. Validation: [ ] object + reasons + PG row + metric.

**M024 Validators** — Objective: rule engine 3.5. Files: `akl/ingestion/validators.py`. Deps: M018. Validation: [ ] each rule tested; [ ] secret patterns.

**M025 Markdown parser** — Files: `akl/ingestion/parsers/markdown.py`. Deps: M018. Validation: [ ] golden snapshots for headings/code/tables/lists/frontmatter/MDX.

**M026 HTML parser** — Files: `akl/ingestion/parsers/html.py, boilerplate.py`. Deps: M025. Validation: [ ] boilerplate removal fixtures; [ ] canonical URL; [ ] code blocks.

**M027 PDF parser** — Files: `akl/ingestion/parsers/pdf.py`. Deps: M018. Validation: [ ] heading detection; [ ] header/footer removal; [ ] table extraction; [ ] image-only page flag; [ ] encrypted/corrupt → codes.

**M028 RST and code parsers** — Files: `parsers/rst.py, parsers/code.py` (tree-sitter). Deps: M026. Validation: [ ] symbols extracted for py/ts/go.

**M029 Silver writer + dedup** — Objective: `silver.documents` write, status transitions, SimHash ledger. Files: `akl/ingestion/silver_writer.py, dedup.py`. Deps: M022, M021. Validation: [ ] near-dup marked; [ ] canonical selection rule.

**M030 PDF & Markdown connectors** — Files: `connectors/pdf.py, connectors/markdown.py`. Deps: M020. Validation: [ ] incremental discover on mtime; [ ] path rules → security level.

**M031 HTML connector** — Files: `connectors/html.py`. Deps: M030. Validation: [ ] robots honoured; [ ] conditional GET; [ ] rate limit.

**M032 GitHub connector** — Objective: API mode + clone fallback + snapshot diff + deletion events. Files: `connectors/github.py`, `bronze/github_snapshots` writer. Deps: M030. Validation: [ ] unchanged head → zero items; [ ] diff add/mod/del; [ ] rate limit sleep.

**M033 Ingestion pipeline service** — Objective: `akl.pipelines.ingestion` functions used by DAG/CLI: discover, fetch, parse_shard, dedup, gate. Files: `akl/pipelines/ingestion.py, akl/cli/ingest.py`. Deps: M022–M032. Validation: [ ] `akl-cli ingest run` seeds Silver from `examples/docs`.

## P3 — Chunking

**M034 Chunk config and tokenizer** — Files: `akl/chunking/config.py, tokenizer.py, configs/chunking.yaml`. Deps: M002. Validation: [ ] config hash stable; [ ] token counts match HF tokenizer.

**M035 Sentence/clause splitting** — Files: `akl/chunking/sentences.py`. Validation: [ ] pysbd edge cases.

**M036 Structural pass** — Files: `akl/chunking/structural.py`. Deps: M018, M034. Validation: [ ] sections never cross sibling headings.

**M037 Code and table splitters** — Files: `code_splitter.py, table_splitter.py`. Deps: M036. Validation: [ ] header repeat; [ ] tree-sitter boundaries; [ ] transpose wide tables.

**M038 Merge/split algorithms** — Files: `merge_split.py`. Deps: M035. Validation: [ ] hypothesis: no chunk > max; min merge behaviour.

**M039 Semantic pass** — Files: `semantic.py` (uses embedding provider from M043 behind interface; stub in tests). Deps: M038. Validation: [ ] boundary at cosine drop.

**M040 Identity, quality, models** — Files: `identity.py, quality.py, models.py`. Deps: M034. Validation: [ ] key/checksum/id stability tests per 4.10 table; [ ] quality formula.

**M041 Chunking engine** — Files: `engine.py`. Deps: M036–M040. Validation: [ ] worked example 4.15 reproduces 4 chunks; [ ] determinism.

**M042 Incremental chunk update + pipeline service** — Files: `incremental.py, akl/pipelines/chunking.py, akl/cli/chunk.py, sql/gold/retrieval_units.sql`. Deps: M041, M014. Validation: [ ] diff classification tests; [ ] no-change run writes zero rows; [ ] Gold retrieval_units refreshed for touched partitions.

## P4 — Embedding & Vector

**M043 Embedding provider (BGE ONNX)** — Files: `akl/embedding/provider.py, bge.py, configs/embedding.yaml`, model download with checksum to `/models`. Deps: M002. Validation: [ ] dim 384, unit norm; [ ] parity torch/ONNX; [ ] query instruction applied.

**M044 Batcher and cache** — Files: `batcher.py, cache.py`. Deps: M009, M043. Validation: [ ] length sort; [ ] hit/miss; [ ] eviction query.

**M045 Gold embeddings writer + jobs + MLflow** — Files: `writer.py, jobs.py`. Deps: M013, M044. Validation: [ ] Parquet rows; [ ] `embedding_jobs`; [ ] MLflow run created.

**M046 Embedding pipeline service** — Files: `akl/pipelines/embedding.py, akl/cli/embed.py, sql/gold/embedding_backlog.sql`. Deps: M045. Validation: [ ] coverage 1.0 after run; [ ] second run zero generation.

**M047 Qdrant client + schema + alias** — Files: `akl/embedding/qdrant/{client,schema}.py, configs (Appendix D)`. Deps: M007. Validation: [ ] idempotent ensure; [ ] payload indexes verified; [ ] alias swap.

**M048 Qdrant reconciler** — Files: `qdrant/reconciler.py`. Deps: M047, M046. Validation: [ ] seeded drift corrected; [ ] counts equal; [ ] `AKL-E5020` on mismatch.

**M049 Blue/green reindex** — Files: `qdrant/reindex.py`. Deps: M048. Validation: [ ] alias points to new collection; [ ] old deleted after grace.

**M050 BM25 tokenizer/builder/index** — Files: `akl/embedding/bm25/*.py`. Deps: M013. Validation: [ ] identifier splitting; [ ] filter; [ ] serialise/load; [ ] retention of 3 versions.

**M051 Qdrant sync pipeline service** — Files: `akl/pipelines/qdrant_sync.py, akl/cli/qdrant.py, akl/cli/bm25.py`. Deps: M048–M050. Validation: [ ] `akl-cli qdrant sync` then `diff` = 0.

## P5 — RAG

**M052 Query normalisation and spell** — Files: `akl/rag/query/normalize.py, spell.py`, vocab builder. Deps: M050. Validation: [ ] protected tokens untouched; [ ] corrections only for OOV.

**M053 Intent and entities** — Files: `intent.py, entities.py, configs/intents.yaml`. Deps: M043. Validation: [ ] labelled set accuracy ≥ 0.85; [ ] gazetteer extraction.

**M054 Filters and principal injection** — Files: `filters.py` + `akl/security/rbac.py` predicate compiler. Deps: M053. Validation: [ ] truth table → Qdrant filter snapshot.

**M055 Dense/sparse/fusion/rerank** — Files: `akl/rag/retrieval/*.py`. Deps: M047, M050, M054. Validation: [ ] RRF math; [ ] concurrency; [ ] reranker parity; [ ] confidence gate.

**M056 Context builder and prompt** — Files: `context_builder.py, prompt.py, configs/prompts/*`. Deps: M055. Validation: [ ] dedupe; [ ] ordering; [ ] budget truncation at sentence.

**M057 LLM providers** — Files: `akl/rag/llm/{provider,openai_compat,extractive}.py`. Deps: M056. Validation: [ ] streaming mock; [ ] timeout → fallback.

**M058 Citations and guards** — Files: `citations.py, guards.py`. Deps: M057. Validation: [ ] marker parsing; [ ] invalid removal; [ ] uncited ratio flag; [ ] unsupported token flag.

**M059 Memory and rewrite** — Files: `memory.py, query/rewrite.py, prompts/summarize_v1.md, rewrite_v1.md`. Deps: M009, M057. Validation: [ ] summary trigger; [ ] rule-based rewrite fallback.

**M060 RAG service + formatter + trace persistence** — Files: `service.py, formatter.py, context.py`. Deps: M052–M059. Validation: [ ] response schema 6.11; [ ] trace row persisted; [ ] insufficient_evidence path.

## P6 — API

**M061 App factory, middleware, lifespan** — Files: `akl/api/main.py, deps.py, middleware/*`. Deps: M060. Validation: [ ] request id echoed; [ ] warm-up before ready.

**M062 Auth (JWT, API keys, service tokens)** — Files: `akl/security/auth/*`, `akl-cli auth mint-token`. Deps: M009. Validation: [ ] 401/403 matrix; [ ] key hashing.

**M063 Rate limiting** — Files: `akl/security/ratelimit.py, middleware/ratelimit.py`. Deps: M062. Validation: [ ] 429 + Retry-After.

**M064 Health router** — Files: `routers/health.py`. Deps: M061. Validation: [ ] ready false when Qdrant version mismatch.

**M065 Documents router (upload, list, detail, chunks, delete, restore)** — Files: `routers/documents.py, schemas/documents.py, governance/deletion.py`. Deps: M022, M062. Validation: [ ] 202/200 paths; [ ] dedup flag; [ ] idempotency key; [ ] cascade delete test.

**M066 Sources router** — Files: `routers/sources.py`. Deps: M032. Validation: [ ] trigger 202; [ ] ownership check.

**M067 Search router** — Files: `routers/search.py, schemas/search.py`. Deps: M060. Validation: [ ] modes; [ ] filters; [ ] leakage test.

**M068 Chat router + SSE** — Files: `routers/chat.py, conversations.py, streaming.py, schemas/chat.py`. Deps: M060. Validation: [ ] event order; [ ] partial marker buffering; [ ] 403 foreign conversation.

**M069 Admin routers** — Files: `routers/admin/*.py, schemas/admin.py`. Deps: M049, M051, M023. Validation: [ ] reindex job progress; [ ] reload-index hot swap; [ ] permissions patch propagates to Qdrant payload.

**M070 OpenAPI export + docs** — Files: `scripts/export_openapi.py`. Deps: M061–M069. Validation: [ ] `openapi.json` committed artefact; [ ] operationIds unique.

## P7 — Airflow

**M071 Airflow image and init** — Files: `docker/airflow.Dockerfile, airflow/config/*, airflow-init service`. Deps: M007. Validation: [ ] webserver healthy; [ ] pools imported.

**M072 Airflow plugin helpers** — Files: `airflow/plugins/akl_airflow/*`. Deps: M005, M006. Validation: [ ] failure callback emits metric + log.

**M073 DAG akl_ingestion** — Files: `airflow/dags/akl_ingestion.py, configs/dags/ingestion.yaml`. Deps: M033, M072. Validation: [ ] `airflow dags test` on seed; [ ] gate fails on forced quarantine ratio; [ ] Dataset published.

**M074 DAG akl_chunking** — Deps: M042, M073. Validation: [ ] triggered by Dataset; [ ] regression gate.

**M075 DAG akl_embedding** — Deps: M046, M074. Validation: [ ] mapped shards; [ ] coverage gate; [ ] MLflow run.

**M076 DAG akl_qdrant_sync** — Deps: M051, M075. Validation: [ ] sensor; [ ] verify counts; [ ] API reload called.

**M077 DAG akl_maintenance** — Deps: M015, M016, M048; retention/eviction/backup tasks (`governance/retention.py`). Validation: [ ] retention idempotent; [ ] backup objects present; [ ] eval run logged.

**M078 DAG integrity tests** — Files: `tests/airflow/test_dag_integrity.py`. Deps: M073–M077. Validation: [ ] no import errors; [ ] every task has `execution_timeout`.

## P8 — Observability

**M079 Prometheus/alertmanager/pushgateway config** — Files: `observability/prometheus/*, alertmanager/*`. Deps: M005. Validation: [ ] targets up; [ ] alert rules load.

**M080 Grafana dashboards as code** — Files: `observability/grafana/**`. Deps: M079. Validation: [ ] 8 dashboards provisioned; [ ] panels query existing metrics only (lint script).

**M081 OTel collector config** — Files: `observability/otel/otel-collector.yml`. Deps: M006. Validation: [ ] traces received (debug exporter).

**M082 Lineage** — Files: `akl/observability/lineage.py`, admin lineage endpoint. Deps: M013, M069. Validation: [ ] document lineage graph resolves Bronze→Qdrant.

**M083 Cost accounting** — Files: `akl/observability/cost.py`. Deps: M057. Validation: [ ] LLM cost metric increments.

**M084 Runbooks and ADR docs** — Files: `docs/runbooks/RB-01..10.md, docs/adr/*.md, docs/mkdocs.yml`. Deps: M079. Validation: [ ] each alert links a runbook.

## P9 — Security & Governance

**M085 Audit logging** — Files: `akl/security/audit.py`; wire into routers. Deps: M062. Validation: [ ] restricted_read audited; [ ] admin actions audited.

**M086 Secrets files & prod compose** — Files: `docker-compose.prod.yml, docker/traefik/*, secrets/README.md`. Deps: M007. Validation: [ ] TLS termination; [ ] no host ports except proxy.

**M087 PII scanner and policy** — Files: `akl/governance/pii.py`, `pii_mentions` table. Deps: M024. Validation: [ ] flag/redact/quarantine modes.

**M088 GDPR endpoints and hard delete** — Files: `akl/governance/gdpr.py`, extend deletion. Deps: M065, M087. Validation: [ ] hard delete purges cache, Gold rows, Bronze object; [ ] citations replaced.

**M089 Permissions propagation** — Files: `akl/governance/permissions.py`. Deps: M069. Validation: [ ] Gold rows rewritten; [ ] Qdrant payload updated; [ ] BM25 patched.

**M090 Threat-model review checklist** — Files: `docs/security/threat_model.md`. Deps: M085–M089. Validation: [ ] each STRIDE row maps to implemented control + test.

## P10 — Testing & Evaluation

**M091 Test fixtures and seed corpus** — Files: `examples/docs/*, tests/fixtures/*, scripts/seed.sh`. Deps: M033. Validation: [ ] 12-doc pipeline fixture per 12.4.

**M092 Unit suite completion to ≥ 85%** — Files: `tests/unit/**`. Deps: all P2–P6. Validation: [ ] coverage report.

**M093 Component suite** — Files: `tests/component/*`. Deps: M009, M013, M047, M050.

**M094 Pipeline e2e + incremental no-op + deletion + reindex tests** — Files: `tests/integration/*`. Deps: M073–M077. Validation: [ ] zero writes on rerun asserted via metrics and file listing.

**M095 API contract suite + leakage** — Files: `tests/api/*`. Deps: M061–M069.

**M096 Synthetic QA generation** — Files: `akl/eval/synth.py, datasets.py`, `akl-cli eval generate`. Deps: M057, M042. Validation: [ ] `gold/eval/qa_pairs/version=v1` written; [ ] curated file loaded.

**M097 Eval metrics and runner + MLflow + calibration** — Files: `akl/eval/{metrics,runner,judge,calibrate}.py`. Deps: M096, M060. Validation: [ ] recall/MRR/nDCG match reference implementation on toy data; [ ] threshold calibration output.

**M098 Eval thresholds test + ablations** — Files: `tests/eval/test_retrieval_thresholds.py`. Deps: M097. Validation: [ ] hybrid ≥ dense-only and ≥ sparse-only on seed set.

**M099 Load tests and benchmark harness** — Files: `tests/load/locustfile.py, akl/cli/bench.py, docs/benchmarks/`. Deps: M067, M068. Validation: [ ] p95 report generated.

## P11 — CI/CD & Release

**M100 Pre-commit + lint config** — Files: `.pre-commit-config.yaml`, ruff/mypy/sqlfluff/yamllint configs. Deps: M001. Validation: [ ] `make lint` clean.

**M101 ci.yml** — Deps: M092, M093, M095, M078. Validation: [ ] green on PR.

**M102 integration.yml + eval.yml nightly** — Deps: M094, M098.

**M103 Dockerfiles multi-stage, multi-arch** — Files: `docker/api.Dockerfile, airflow.Dockerfile`. Deps: M007. Validation: [ ] image size budgets; [ ] non-root.

**M104 release.yml (GHCR, SBOM, cosign, changelog)** — Deps: M103. Validation: [ ] tag `v0.1.0` produces release.

**M105 deploy.yml + VM bootstrap** — Files: `deploy/vm/*`. Deps: M086, M104. Validation: [ ] staging deploy + `akl-cli verify --all`.

**M106 security.yml + dependabot** — Deps: M101.

**M107 docs.yml (MkDocs site)** — Deps: M084, M070.

## P12 — Hardening

**M108 Backup/restore automation and drill** — Files: `akl/pipelines/maintenance.py (backup tasks), docs/runbooks/RB-11-restore.md`. Deps: M077. Validation: [ ] restore drill from backups reproduces `drift == 0`.

**M109 Runtime retrieval config hot reload** — Deps: M069. Validation: [ ] PUT config applied within 10 s without restart.

**M110 Reembed flow end-to-end** — Deps: M049, M069. Validation: [ ] new `embedding_version` backfilled; [ ] promote swaps alias; [ ] old partition retired by maintenance.

**M111 Performance pass** — Objective: meet NFR-01/02/04 on reference laptop: ONNX INT8, thread tuning, query cache, adaptive `hnsw_ef`. Deps: M099. Validation: [ ] benchmark rows recorded.

**M112 Chaos tests** — Objective: kill Qdrant / LLM / Postgres during requests and DAG runs; assert degraded modes and codes (6.12, 3.9). Files: `tests/integration/test_chaos.py`. Deps: M094. Validation: [ ] sparse-only degrade; [ ] extractive fallback; [ ] DAG retry safe.

**M113 Free-cloud deployment** — Deps: M105. Validation: [ ] public instance answers seed queries with citations; [ ] TLS valid.

**M114 Documentation freeze and v1.0.0 release** — Deps: all. Validation: [ ] every FR/NFR row maps to tests; [ ] CHANGELOG; [ ] tag `v1.0.0`.

---

# Appendix A — SQL Schemas (PostgreSQL `akl` database)

Notation: PK = primary key; FK = foreign key; IDX = index. All tables have `created_at TIMESTAMPTZ DEFAULT now()` and, where mutable, `updated_at`. UUIDs are `uuid` type. Large tables are declared with partition keys for Enterprise Scale (declarative partitioning in Postgres is used only for `audit_log` in MVP).

## A.1 `documents`

| Column | Type | Constraints |
|---|---|---|
| `document_id` | uuid | PK |
| `canonical_source_uri` | text | UNIQUE, NOT NULL |
| `source_type` | text | NOT NULL, CHECK in (pdf,markdown,html,github) |
| `connector_id` | text | NOT NULL |
| `title` | text | |
| `status` | text | NOT NULL, CHECK in (bronze,silver,gold,quarantined,deleting,deleted) |
| `current_version_id` | uuid | FK → document_versions |
| `latest_content_sha256` | char(64) | |
| `security_level` | text | NOT NULL, CHECK in (public,internal,restricted) |
| `allowed_groups` | text[] | NOT NULL DEFAULT '{}' |
| `is_duplicate_of` | uuid | FK → documents |
| `pii_types` | text[] | |
| `metadata` | jsonb | |
| `deleted_at` | timestamptz | |
IDX: `(source_type, status)`, `(connector_id)`, GIN `(allowed_groups)`, `(updated_at)`.

## A.2 `document_versions`

| Column | Type | Constraints |
|---|---|---|
| `document_version_id` | uuid | PK |
| `document_id` | uuid | FK → documents NOT NULL |
| `content_sha256` | char(64) | NOT NULL |
| `text_sha256` | char(64) | |
| `bronze_object_key` | text | NOT NULL |
| `parser_name`, `parser_version` | text | |
| `silver_partition` | text | e.g. `source_type=github/ingest_date=2026-09-04` |
| `quality_score` | real | |
| `quality_flags` | text[] | |
| `language` | varchar(8) | |
| `word_count` | int | |
| `run_id` | text | |
| `fetched_at`, `parsed_at` | timestamptz | |
UNIQUE `(document_id, content_sha256, parser_version)`; IDX `(content_sha256)`, `(run_id)`.

## A.3 `chunks` (current-state index; full history in Silver Parquet)

| Column | Type | Constraints |
|---|---|---|
| `chunk_id` | uuid | PK |
| `chunk_key` | char(40) | NOT NULL |
| `lineage_id` | uuid | NOT NULL |
| `chunk_checksum` | char(64) | NOT NULL |
| `embedded_text_sha256` | char(64) | |
| `document_id` | uuid | FK NOT NULL |
| `document_version_id` | uuid | FK NOT NULL |
| `chunk_index` | int | NOT NULL |
| `chunk_type` | text | NOT NULL |
| `token_count` | int | |
| `quality_score` | real | |
| `security_level` | text | NOT NULL |
| `allowed_groups` | text[] | NOT NULL DEFAULT '{}' |
| `chunker_version` | text | NOT NULL |
| `chunk_config_hash` | char(16) | NOT NULL |
| `is_current` | boolean | NOT NULL DEFAULT true |
| `is_deleted` | boolean | NOT NULL DEFAULT false |
| `embedding_status` | text | CHECK in (pending,embedded,failed) |
IDX: `(document_id, is_current)`, `(lineage_id)`, `(chunk_checksum)`, `(embedded_text_sha256)`, `(embedding_status) WHERE is_current`, `(document_version_id, chunker_version, chunk_config_hash)`. Partition key (Enterprise): hash `(document_id)`.

## A.4 `embedding_cache`

| Column | Type | Constraints |
|---|---|---|
| `embedded_text_sha256` | char(64) | PK part |
| `model_id` | text | PK part |
| `model_version` | text | PK part |
| `dim` | smallint | NOT NULL |
| `vector` | bytea | NOT NULL (float32 LE, dim×4 bytes) |
| `hit_count` | int | DEFAULT 0 |
| `last_hit_at` | timestamptz | |
IDX `(last_hit_at)` for eviction.

## A.5 `embedding_jobs`, `embedding_backlog`

`embedding_jobs(job_id uuid PK, run_id text, embedding_version text, shard int, chunks_total int, cache_hits int, generated int, failed int, started_at, finished_at, throughput_cps real, mlflow_run_id text)`; IDX `(run_id)`.
`embedding_backlog(run_id text, chunk_id uuid, shard int, PRIMARY KEY (run_id, chunk_id))`; IDX `(run_id, shard)`.

## A.6 `qdrant_sync_ops`

`(run_id text, op text CHECK in (upsert,delete), chunk_id uuid, applied boolean DEFAULT false, PRIMARY KEY (run_id, op, chunk_id))`.

## A.7 `pipeline_runs`, `task_runs`

`pipeline_runs(run_id text PK, dag_id text, correlation_id text, state text, started_at, finished_at, conf jsonb, gold_snapshot_id text)`; IDX `(dag_id, started_at DESC)`.
`task_runs(id bigserial PK, run_id text FK, task_id text, map_index int, try_number int, state text, started_at, finished_at, rows_in bigint, rows_out bigint, metrics jsonb)`; IDX `(run_id)`.

## A.8 `lineage_edges`

`(id bigserial PK, run_id text, task_id text, input_dataset text, input_partition text, output_dataset text, output_partition text, rows_in bigint, rows_out bigint, created_at)`; IDX `(output_dataset, output_partition)`, `(run_id)`.

## A.9 `connector_state`

`(connector_id text PK, connector_name text, state jsonb NOT NULL, last_run_id text, last_success_at timestamptz, documents_count int, updated_at)`.

## A.10 `quarantine_items`

`(quarantine_id uuid PK, document_id uuid, content_sha256 char(64), source_type text, error_code text, stage text, detail text, object_key text, run_id text, status text CHECK in (open,retried,dismissed), created_at, resolved_at)`; IDX `(status, created_at)`, `(error_code)`.

## A.11 `users`, `api_keys`

`users(user_id uuid PK, subject text UNIQUE, email text, groups text[], security_levels text[], roles text[], disabled boolean, created_at, last_login_at)`.
`api_keys(key_id uuid PK, prefix char(8) UNIQUE, key_hash text NOT NULL, name text, owner_user_id uuid FK, scopes text[], groups text[], security_levels text[], expires_at, last_used_at, revoked_at)`; IDX `(prefix)`.

## A.12 `conversations`, `messages`, `answer_citations`, `retrieval_traces`

`conversations(conversation_id uuid PK, principal_id text NOT NULL, summary text, summary_tokens int, turn_count int, created_at, updated_at, expires_at)`; IDX `(principal_id, updated_at)`, `(expires_at)`.
`messages(message_id uuid PK, conversation_id uuid FK, turn int, role text, content text, rewritten_query text, token_count int, trace_id text, request_id text, mode text, confidence real, flags text[], created_at)`; IDX `(conversation_id, turn)`.
`answer_citations(id bigserial PK, message_id uuid FK, citation_index int, chunk_id uuid, lineage_id uuid, document_id uuid, locator text, score real)`; IDX `(message_id)`, `(chunk_id)`.
`retrieval_traces(trace_id text PK, request_id text, principal_id text, query text (nullable; stored only if AKL_STORE_QUERY_TEXT), intent text, filters jsonb, dense_ids uuid[], sparse_ids uuid[], fused_ids uuid[], reranked jsonb, confidence real, gold_snapshot_id text, timings jsonb, created_at)`; IDX `(created_at)`.

## A.13 `audit_log` (partitioned by month on `ts`)

`(id bigserial, ts timestamptz NOT NULL, principal_id text, action text NOT NULL, resource_type text, resource_id text, request_id text, ip inet, user_agent text, outcome text, details jsonb, PRIMARY KEY (id, ts)) PARTITION BY RANGE (ts)`; IDX `(principal_id, ts)`, `(action, ts)`, `(resource_id)`.

## A.14 `runtime_config`, `idempotency_keys`, `admin_jobs`, `pii_mentions`, `lakehouse_schema_versions`, `lakehouse_files`, `rate_limit_buckets`

- `runtime_config(key text PK, value jsonb, updated_by text, updated_at)`.
- `idempotency_keys(key text PK, principal_id text, request_hash text, response jsonb, created_at)`; IDX `(created_at)` (24 h purge).
- `admin_jobs(job_id uuid PK, type text, status text, progress real, params jsonb, result jsonb, started_by text, started_at, finished_at, error_code text)`.
- `pii_mentions(id bigserial PK, document_id uuid, chunk_id uuid, pii_type text, value_hash text)`; IDX `(value_hash)`, `(document_id)`.
- `lakehouse_schema_versions(dataset text, schema_version text, pyarrow_schema_json jsonb, first_written_at, PRIMARY KEY (dataset, schema_version))`.
- `lakehouse_files(dataset text, partition text, object_key text PRIMARY KEY, rows bigint, bytes bigint, run_id text, is_active boolean, created_at)`; IDX `(dataset, partition, is_active)`.
- `rate_limit_buckets(principal_id text, route_class text, tokens real, updated_at, PRIMARY KEY (principal_id, route_class))`.

## A.15 ER Diagram

```mermaid
erDiagram
    documents ||--o{ document_versions : has
    documents ||--o{ chunks : has
    document_versions ||--o{ chunks : produces
    chunks }o--o| embedding_cache : "embedded_text_sha256"
    users ||--o{ api_keys : owns
    conversations ||--o{ messages : has
    messages ||--o{ answer_citations : has
    messages ||--o| retrieval_traces : trace_id
    answer_citations }o--|| chunks : chunk_id
    pipeline_runs ||--o{ task_runs : has
    pipeline_runs ||--o{ lineage_edges : has
    pipeline_runs ||--o{ qdrant_sync_ops : has
    pipeline_runs ||--o{ embedding_jobs : has
    documents ||--o{ quarantine_items : may_have
    documents ||--o{ pii_mentions : has
```

---

# Appendix B — Environment Variables

All application variables use the `AKL_` prefix and are validated by `akl.config.Settings`. Any variable ending in `_FILE` overrides its counterpart by reading the file contents. Types: `str`, `int`, `float`, `bool` (`true/false`), `enum`, `list` (comma-separated), `secret`.

## B.1 Core

| Variable | Type | Default | Purpose | Example |
|---|---|---|---|---|
| `AKL_ENV` | enum(dev,staging,prod) | `dev` | Environment; gates unsafe options | `prod` |
| `AKL_SERVICE_NAME` | str | `akl-api` | Logged/traced service name | `akl-airflow` |
| `AKL_LOG_LEVEL` | enum | `INFO` | Log level | `DEBUG` |
| `AKL_LOG_SAMPLE_DEBUG` | bool | `false` | Sample 1% of debug logs | |
| `AKL_LOG_QUERIES` | bool | `false` | Log raw query text (dev only) | |
| `AKL_STORE_QUERY_TEXT` | bool | `false` | Persist query text in `retrieval_traces` | |
| `AKL_ORG_NAME` | str | `Example Org` | Injected into system prompt | `DevX Labs` |
| `AKL_CONFIG_DIR` | str | `/app/configs` | Location of YAML configs | |
| `AKL_MODELS_DIR` | str | `/models` | Model cache volume | |
| `AKL_TMP_DIR` | str | `/tmp/akl` | Parser scratch | |

## B.2 PostgreSQL

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_DB_HOST` | str | `postgres` | |
| `AKL_DB_PORT` | int | `5432` | |
| `AKL_DB_NAME` | str | `akl` | |
| `AKL_DB_USER` | str | `akl_api` (API) / `akl_pipeline` (Airflow) | Role per service |
| `AKL_DB_PASSWORD` / `_FILE` | secret | — | |
| `AKL_DB_POOL_MIN` / `AKL_DB_POOL_MAX` | int | `5` / `20` | asyncpg pool |
| `AKL_DB_SSLMODE` | enum | `disable` (dev) / `require` (prod) | |
| `POSTGRES_USER`, `POSTGRES_PASSWORD(_FILE)` | secret | — | Container superuser (compose) |

## B.3 Object Storage (MinIO / S3)

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_S3_ENDPOINT` | str | `http://minio:9000` | S3 API endpoint |
| `AKL_S3_REGION` | str | `us-east-1` | |
| `AKL_S3_BUCKET` | str | `akl-lakehouse` | Single bucket |
| `AKL_S3_ACCESS_KEY` / `_FILE` | secret | — | |
| `AKL_S3_SECRET_KEY` / `_FILE` | secret | — | |
| `AKL_S3_USE_SSL` | bool | `false` | |
| `AKL_S3_PATH_STYLE` | bool | `true` | Required for MinIO |
| `AKL_BRONZE_OBJECT_LOCK` | bool | `false` | Enable governance-mode lock |
| `AKL_BRONZE_RETENTION_DAYS` | int | `365` | |
| `AKL_LAKEHOUSE_USE_FILE_MANIFEST` | bool | `false` | Read via `lakehouse_files` instead of glob |
| `AKL_PARQUET_COMPRESSION` | enum | `ZSTD` | |
| `AKL_PARQUET_ZSTD_LEVEL` | int | `3` | |
| `AKL_COMPACT_MIN_FILES` | int | `8` | Compaction trigger |
| `AKL_COMPACT_SMALL_FILE_MB` | int | `32` | |
| `AKL_DUCKDB_MEMORY_LIMIT` | str | `4GB` | DuckDB `memory_limit` |
| `AKL_DUCKDB_THREADS` | int | `4` | |
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD(_FILE)` | secret | — | Container (compose) |

## B.4 Qdrant

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_QDRANT_URL` | str | `http://qdrant:6333` | |
| `AKL_QDRANT_GRPC_PORT` | int | `6334` | |
| `AKL_QDRANT_PREFER_GRPC` | bool | `true` | |
| `AKL_QDRANT_API_KEY` / `_FILE` | secret | — | Optional |
| `AKL_QDRANT_COLLECTION_ALIAS` | str | `kb_chunks` | API queries alias |
| `AKL_QDRANT_COLLECTION` | str | `kb_chunks_v1` | Physical collection for sync |
| `AKL_QDRANT_HNSW_M` | int | `16` | |
| `AKL_QDRANT_HNSW_EF_CONSTRUCT` | int | `128` | |
| `AKL_QDRANT_HNSW_EF` | int | `128` | Search-time default |
| `AKL_QDRANT_ON_DISK_PAYLOAD` | bool | `true` | |
| `AKL_QDRANT_UPSERT_BATCH` | int | `512` | |
| `AKL_QDRANT_DELETE_BATCH` | int | `1000` | |
| `AKL_QDRANT_SCROLL_PAGE` | int | `10000` | Reconciler page size |
| `AKL_REINDEX_GRACE_HOURS` | int | `24` | Old collection retention after alias swap |

## B.5 Embedding

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_EMBED_MODEL_ID` | str | `BAAI/bge-small-en-v1.5` | |
| `AKL_EMBED_MODEL_VERSION` | str | `1.5` | |
| `AKL_EMBED_MODEL_SHA256` | str | — | Checksum of ONNX file; verified on load |
| `AKL_EMBEDDING_VERSION` | str | `bge-small-en-v1.5__1.5__384` | Derived if unset |
| `AKL_EMBED_DIM` | int | `384` | |
| `AKL_EMBED_DEVICE` | enum(auto,cpu,cuda) | `auto` | |
| `AKL_EMBED_BATCH_SIZE` | int | `64` | |
| `AKL_EMBED_THREADS` | int | `cores−1` | ONNX intra-op |
| `AKL_EMBED_ONNX_INT8` | bool | `false` | |
| `AKL_EMBED_QUERY_INSTRUCTION` | str | `Represent this sentence for searching relevant passages: ` | |
| `AKL_EMBED_TASK_SHARDS` | int | `4` | Dynamic-mapped shards |
| `AKL_EMBEDDING_CACHE_TTL_DAYS` | int | `180` | |
| `AKL_EMBEDDING_RETIRE_DAYS` | int | `30` | Old version partition retention |
| `AKL_RERANK_MODEL_ID` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `AKL_RERANK_ENABLED` | bool | `true` | |
| `AKL_RERANK_THREADS` | int | `2` | |
| `AKL_RERANK_ONNX_INT8` | bool | `true` | |

## B.6 Chunking

| Variable | Type | Default |
|---|---|---|
| `AKL_CHUNKER_VERSION` | str | `1.0.0` |
| `AKL_CHUNK_TARGET_TOKENS` | int | `320` |
| `AKL_CHUNK_MAX_TOKENS` | int | `448` |
| `AKL_CHUNK_MIN_TOKENS` | int | `64` |
| `AKL_CHUNK_OVERLAP_TOKENS` | int | `48` |
| `AKL_CHUNK_SEMANTIC_ENABLED` | bool | `true` |
| `AKL_CHUNK_SEMANTIC_THRESHOLD` | float | `0.25` |
| `AKL_CHUNK_CODE_MAX_TOKENS` | int | `400` |
| `AKL_CHUNK_TABLE_MAX_TOKENS` | int | `400` |
| `AKL_CHUNK_CONTEXT_PREFIX_TOKENS` | int | `40` |
| `AKL_CHUNK_EMIT_SECTION_PARENTS` | bool | `false` |
| `AKL_CHUNK_QUALITY_MIN` | float | `0.30` |
| `AKL_DOC_QUALITY_MIN` | float | `0.35` |

## B.7 Ingestion

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_PARSER_VERSION` | str | `1.0.0` | Bumping forces reparse |
| `AKL_MIN_DOC_BYTES` / `AKL_MAX_DOC_BYTES` | int | `64` / `52428800` | |
| `AKL_MAX_UPLOAD_MB` | int | `50` | API uploads |
| `AKL_FETCH_CONCURRENCY` | int | `8` | Per fetch task |
| `AKL_PARSE_SHARDS` | int | `4` | |
| `AKL_PARSE_TIMEOUT_S` | int | `120` | |
| `AKL_PARSE_MAX_RSS_MB` | int | `2048` | Parser subprocess limit |
| `AKL_PDF_MAX_PAGES` | int | `2000` | |
| `AKL_PDF_TRY_EMPTY_PASSWORD` | bool | `true` | |
| `AKL_PDF_INBOX_PATH` | str | `s3://akl-lakehouse/inbox/pdf/` | |
| `AKL_HTML_USER_AGENT` | str | `AKL-Crawler/1.0 (+contact)` | |
| `AKL_HTML_RPS_PER_HOST` | float | `2` | |
| `AKL_GITHUB_TOKEN` / `_FILE` | secret | — | |
| `AKL_GITHUB_API_URL` | str | `https://api.github.com` | GHES support |
| `AKL_GITHUB_MODE` | enum(api,clone,auto) | `auto` | |
| `AKL_ALLOW_SECRET_LIKE_CONTENT` | bool | `false` | |
| `AKL_PII_POLICY` | enum(flag,redact,quarantine) | `flag` | |
| `AKL_GATE_QUARANTINE_RATIO` | float | `0.25` | Ingestion gate |
| `AKL_QUARANTINE_RETENTION_DAYS` | int | `90` | |

## B.8 Retrieval and RAG

| Variable | Type | Default |
|---|---|---|
| `AKL_QUERY_MAX_CHARS` | int | `2000` |
| `AKL_QUERY_SPELL_DUAL` | bool | `true` |
| `AKL_RETRIEVAL_DENSE_K` | int | `50` |
| `AKL_RETRIEVAL_SPARSE_K` | int | `50` |
| `AKL_RETRIEVAL_FUSED_K` | int | `40` |
| `AKL_RRF_K` | int | `60` |
| `AKL_RAG_TOP_K` | int | `8` |
| `AKL_RAG_MIN_CONFIDENCE` | float | `0.35` |
| `AKL_RAG_MIN_CANDIDATES` | int | `2` |
| `AKL_RAG_CONTEXT_TOKENS` | int | `3000` |
| `AKL_RAG_NEIGHBOR_EXPANSION` | int | `0` |
| `AKL_RAG_HISTORY_TURNS` | int | `3` |
| `AKL_RAG_SUMMARY_TRIGGER_TOKENS` | int | `1500` |
| `AKL_RAG_MAX_UNCITED_RATIO` | float | `0.2` |
| `AKL_RAG_STRICT` | bool | `false` |
| `AKL_RAG_NLI_ENABLED` | bool | `false` |
| `AKL_QUERY_CACHE_SIZE` | int | `2048` |
| `AKL_RESULT_CACHE_TTL_S` | int | `0` |
| `AKL_CONVERSATION_TTL_DAYS` | int | `30` |
| `AKL_ALLOW_SPARSE_UNAVAILABLE` | bool | `false` |
| `AKL_BM25_K1` / `AKL_BM25_B` | float | `1.5` / `0.75` |

## B.9 LLM

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AKL_LLM_PROVIDER` | enum(none,openai_compat) | `none` | `none` → extractive |
| `AKL_LLM_BASE_URL` | str | `http://host.docker.internal:11434/v1` | OpenAI-compatible endpoint |
| `AKL_LLM_API_KEY` / `_FILE` | secret | — | |
| `AKL_LLM_MODEL` | str | — | e.g. `llama3.1:8b` |
| `AKL_LLM_TEMPERATURE` | float | `0.1` | |
| `AKL_LLM_MAX_INPUT_TOKENS` | int | `6000` | |
| `AKL_LLM_MAX_OUTPUT_TOKENS` | int | `800` | |
| `AKL_LLM_TIMEOUT_S` | int | `60` | |
| `AKL_LLM_PRICE_INPUT_PER_1K` / `_OUTPUT_PER_1K` | float | `0` | Cost metrics |

## B.10 API and Security

| Variable | Type | Default |
|---|---|---|
| `AKL_API_HOST` / `AKL_API_PORT` | str/int | `0.0.0.0` / `8000` |
| `AKL_API_WORKERS` | int | `2` |
| `AKL_OPENAPI_ENABLED` | bool | `true` (dev) / `false` (prod) |
| `AKL_CORS_ORIGINS` | list | `` |
| `AKL_AUTH_DISABLED` | bool | `false` (only honoured when `AKL_ENV=dev`) |
| `AKL_JWT_ISSUER` | str | `akl-local` |
| `AKL_JWT_AUDIENCE` | str | `akl-api` |
| `AKL_JWT_JWKS_URL` | str | — |
| `AKL_JWT_PRIVATE_KEY_PATH` / `AKL_JWT_PUBLIC_KEY_PATH` | str | `/run/secrets/jwt_private.pem` / `…public.pem` |
| `AKL_JWT_TTL_S` | int | `3600` |
| `AKL_API_KEY_PREFIX` | str | `akl_` |
| `AKL_RATE_LIMIT_RPM` | int | `120` |
| `AKL_RATE_LIMIT_CHAT_RPM` | int | `30` |
| `AKL_AIRFLOW_API_URL` | str | `http://airflow-webserver:8080/api/v1` |
| `AKL_AIRFLOW_API_USER` / `AKL_AIRFLOW_API_PASSWORD(_FILE)` | secret | — |
| `AKL_TRIGGER_DEBOUNCE_S` | int | `60` |

## B.11 Observability

| Variable | Type | Default |
|---|---|---|
| `AKL_METRICS_ENABLED` | bool | `true` |
| `AKL_PUSHGATEWAY_URL` | str | `http://pushgateway:9091` |
| `AKL_OTEL_ENABLED` | bool | `true` |
| `AKL_OTEL_EXPORTER_ENDPOINT` | str | `http://otel-collector:4317` |
| `AKL_OTEL_SAMPLE_RATIO` | float | `0.2` |
| `AKL_MLFLOW_TRACKING_URI` | str | `http://mlflow:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | str | `http://minio:9000` |
| `AKL_ALERT_SLACK_WEBHOOK` / `_FILE` | secret | — |
| `AKL_ALERT_EMAIL_TO`, `AKL_ALERT_SMTP_URL` | str | — |

## B.12 Airflow (native names)

| Variable | Default |
|---|---|
| `AIRFLOW__CORE__EXECUTOR` | `LocalExecutor` |
| `AIRFLOW__CORE__PARALLELISM` | `8` |
| `AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG` | `1` |
| `AIRFLOW__CORE__LOAD_EXAMPLES` | `False` |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | `postgresql+psycopg2://airflow:***@postgres/airflow` |
| `AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL` | `30` |
| `AIRFLOW__METRICS__STATSD_ON` | `True` |
| `AIRFLOW__METRICS__STATSD_HOST` | `statsd-exporter` |
| `AIRFLOW__WEBSERVER__SECRET_KEY(_FILE)` | secret |
| `_AIRFLOW_WWW_USER_USERNAME` / `_PASSWORD` | `admin` / secret |
| `AKL_AIRFLOW_EXECUTOR` | `LocalExecutor` (selects compose profile) |

## B.13 Maintenance and Backup

| Variable | Default |
|---|---|
| `AKL_BACKUP_RETENTION_DAYS` | `14` |
| `AKL_BACKUP_ENCRYPT` | `false` |
| `AKL_BACKUP_MIRROR_TARGET` | — |
| `AKL_SILVER_SUPERSEDED_RETENTION_DAYS` | `180` |

---

# Appendix C — Airflow DAG Config

| DAG | Cron | Dataset triggers | max_active_runs | Default retries | Retry delay | Backoff max | SLA | Catchup |
|---|---|---|---|---|---|---|---|---|
| `akl_ingestion` | `*/30 * * * *` | — | 1 | 2 | 2 m | 20 m | 25 m | False |
| `akl_chunking` | `15 * * * *` | `akl://silver/documents` | 1 | 2 | 2 m | 20 m | 30 m | False |
| `akl_embedding` | `30 * * * *` | `akl://silver/chunks` | 1 | 2 | 3 m | 30 m | 60 m | False |
| `akl_qdrant_sync` | `45 * * * *` | `akl://gold/chunk_embeddings` | 1 | 3 | 1 m | 10 m | 30 m | False |
| `akl_maintenance` | `0 2 * * *` | — | 1 | 1 | 5 m | 5 m | 90 m | False |

## C.1 Per-task Timeouts

| DAG | Task | execution_timeout | Retries override |
|---|---|---|---|
| ingestion | load_connector_configs | 1 m | 1 |
| ingestion | discover[*] | 15 m | |
| ingestion | fetch_to_bronze[*] | 45 m | |
| ingestion | checkpoint_connectors | 2 m | |
| ingestion | list_bronze_backlog | 5 m | |
| ingestion | parse_validate_to_silver[*] | 60 m | 1 |
| ingestion | dedup_pass | 15 m | |
| ingestion | quality_gate | 2 m | 0 |
| ingestion | publish_silver_dataset | 1 m | |
| ingestion | emit_metrics | 2 m | 0 |
| chunking | list_documents_needing_chunks | 5 m | |
| chunking | chunk_documents[*] | 45 m | |
| chunking | aggregate_chunk_diffs | 5 m | |
| chunking | quality_gate | 2 m | 0 |
| chunking | refresh_gold_retrieval_units | 20 m | |
| embedding | warm_model_check | 10 m | 3 |
| embedding | compute_embedding_backlog | 10 m | |
| embedding | embed_shard[*] | 90 m | 2 (batch halving) |
| embedding | merge_and_write_gold | 15 m | |
| embedding | log_mlflow_run | 5 m | |
| embedding | coverage_gate | 2 m | 0 |
| qdrant_sync | qdrant_health_sensor | 10 m (poke 30 s) | |
| qdrant_sync | ensure_collection_and_indexes | 5 m | |
| qdrant_sync | compute_diff | 15 m | |
| qdrant_sync | upsert_points | 45 m | 3 |
| qdrant_sync | delete_points | 15 m | 3 |
| qdrant_sync | verify_counts | 5 m | 0 |
| qdrant_sync | rebuild_bm25_index | 20 m | |
| qdrant_sync | rebuild_vocab | 10 m | |
| qdrant_sync | notify_api_reload | 2 m | 3 |
| maintenance | compact_partitions | 45 m | |
| maintenance | *_retention / eviction / ttl | 15 m each | |
| maintenance | compute_corpus_stats | 15 m | |
| maintenance | run_retrieval_evaluation | 30 m | |
| maintenance | backup_postgres_and_qdrant_snapshot | 30 m | |
| maintenance | vacuum_analyze_postgres | 15 m | |

## C.2 Pools

| Pool | Slots | Used by |
|---|---|---|
| `default_pool` | 8 | all |
| `akl_embedding` | `AKL_EMBED_TASK_SHARDS` (4) | `embed_shard` |
| `akl_github_api` | 2 | `discover`/`fetch` for GitHub connectors |
| `akl_heavy_parse` | 2 | PDF parse shards |

## C.3 Manual `conf` Parameters

| DAG | Key | Effect |
|---|---|---|
| ingestion | `connectors: [ids]` | Restrict to listed connectors |
| ingestion | `document_ids: [uuid]` | Parse only listed (used by upload API) |
| ingestion | `force_reparse: bool` | Ignore `(sha, parser_version)` skip |
| ingestion | `full: bool` | GitHub full tree re-fetch |
| chunking | `document_ids`, `force_rechunk` | |
| embedding | `embedding_version` | Backfill new version |
| qdrant_sync | `reindex: bool`, `target_collection` | Blue/green |
| maintenance | `skip: [task_ids]` | Skip listed tasks |
| all | `correlation_id` | Propagated to logs/traces |

---

# Appendix D — Qdrant Collection Config

## D.1 Collection `kb_chunks_v1`

| Parameter | Value |
|---|---|
| `vectors.size` | 384 |
| `vectors.distance` | `Cosine` |
| `vectors.on_disk` | false (MVP) / true for cold collections (Enterprise) |
| `hnsw_config.m` | 16 |
| `hnsw_config.ef_construct` | 128 |
| `hnsw_config.full_scan_threshold` | 10000 |
| `hnsw_config.on_disk` | false |
| `optimizers_config.default_segment_number` | 2 |
| `optimizers_config.indexing_threshold` | 20000 |
| `optimizers_config.memmap_threshold` | 200000 |
| `on_disk_payload` | true |
| `shard_number` | 1 (MVP) |
| `replication_factor` | 1 (MVP) |
| `write_consistency_factor` | 1 |
| `quantization_config` | null (MVP); `scalar{type: int8, quantile: 0.99, always_ram: true}` (Enterprise) |
| alias | `kb_chunks` |

## D.2 Payload Indexes

| Field | Schema | Notes |
|---|---|---|
| `document_id` | keyword | delete-by-document, dedupe |
| `source_type` | keyword | |
| `security_level` | keyword | mandatory |
| `allowed_groups` | keyword (array) | mandatory |
| `chunk_type` | keyword | |
| `code_language` | keyword | |
| `repo` | keyword | |
| `token_count` | integer | |
| `document_updated_at` | integer (epoch s) | range filters |
| `quality_score` | float | |
| `untrusted` | bool | public HTML de-prioritisation |
| `embedding_version` | keyword | reconciler |

## D.3 Search Parameters

| Param | Default | Notes |
|---|---|---|
| `limit` | 50 | dense_k |
| `hnsw_ef` | 128 | 64 for k≤5, 256 for `precision=high` |
| `exact` | false | true only in eval "ground-truth" mode |
| `with_payload` | include list (payload contract 2.6.1 minus `text` when `include_text=false`) | |
| `with_vectors` | false | |
| `score_threshold` | none | thresholding done post-rerank |

## D.4 Filter Template (security)

```
must:
  - key: security_level, match: { any: <principal.security_levels> }
should (min_should=1):
  - key: allowed_groups, match: { any: <principal.groups> }
  - is_empty: { key: allowed_groups }
```

Hard user filters appended to `must`; soft filters implemented as two-pass (6.2.5).

---

# Appendix E — Docker Compose Specification

## E.1 Services

| Service | Image | Command/Entrypoint notes | Restart | Profiles |
|---|---|---|---|---|
| `postgres` | postgres:16-alpine | init scripts create 3 DBs and roles | unless-stopped | — |
| `minio` | minio/minio | `server /data --console-address :9001` | unless-stopped | — |
| `minio-init` | minio/mc | `docker/minio/init.sh` one-shot | no | — |
| `qdrant` | qdrant/qdrant:v1.12 | env `QDRANT__SERVICE__API_KEY` optional | unless-stopped | — |
| `airflow-init` | akl/airflow | `db migrate; users create; pools import; variables import` | no | — |
| `airflow-scheduler` | akl/airflow | `scheduler` | unless-stopped | — |
| `airflow-webserver` | akl/airflow | `webserver` | unless-stopped | — |
| `airflow-triggerer` | akl/airflow | `triggerer` | unless-stopped | — |
| `airflow-worker` | akl/airflow | `celery worker` | unless-stopped | `celery` |
| `statsd-exporter` | prom/statsd-exporter | mapping file | unless-stopped | — |
| `akl-api` | akl/api | `alembic upgrade head && uvicorn akl.api.main:app` | unless-stopped | — |
| `mlflow` | akl/mlflow | `mlflow server --backend-store-uri postgresql://… --artifacts-destination s3://akl-lakehouse/mlflow` | unless-stopped | — |
| `prometheus` | prom/prometheus | `--config.file … --storage.tsdb.retention.time=30d` | unless-stopped | — |
| `pushgateway` | prom/pushgateway | | unless-stopped | — |
| `alertmanager` | prom/alertmanager | | unless-stopped | — |
| `grafana` | grafana/grafana:11 | provisioning mounted | unless-stopped | — |
| `otel-collector` | otel/opentelemetry-collector-contrib | config mounted | unless-stopped | — |
| `traefik` | traefik:v3 | | unless-stopped | `prod` |
| `loki`, `promtail`, `tempo`, `cadvisor` | official | | unless-stopped | `observability` |

## E.2 Volumes

`akl_pg_data`, `akl_minio_data`, `akl_qdrant_data`, `akl_qdrant_snapshots`, `akl_models`, `akl_airflow_logs`, `akl_prom_data`, `akl_grafana_data`, `akl_tmp`, `akl_traefik_certs` (prod), `akl_loki_data`, `akl_tempo_data` (observability).

## E.3 Networks

| Network | Driver | Internal | Members |
|---|---|---|---|
| `akl-edge` | bridge | no | traefik, akl-api, airflow-webserver, grafana, mlflow, minio (console) |
| `akl-data` | bridge | no (dev) / yes (prod) | everything else + the edge services |
| `akl-egress` | bridge | no | akl-api, airflow-scheduler, airflow-worker (prod only; GitHub/LLM egress) |

## E.4 Health Checks

| Service | Test | Interval / Timeout / Retries / Start |
|---|---|---|
| postgres | `pg_isready -U $POSTGRES_USER` | 10 s / 5 s / 10 / 10 s |
| minio | `curl -f http://localhost:9000/minio/health/live` | 15 s / 5 s / 10 / 10 s |
| qdrant | `curl -f http://localhost:6333/readyz` | 15 s / 5 s / 10 / 20 s |
| akl-api | `curl -f http://localhost:8000/v1/health/ready` | 20 s / 5 s / 10 / 180 s |
| airflow-scheduler | `airflow jobs check --job-type SchedulerJob --hostname "$HOSTNAME"` | 30 s / 10 s / 5 / 60 s |
| airflow-webserver | `curl -f http://localhost:8080/health` | 30 s / 10 s / 5 / 60 s |
| airflow-triggerer | `airflow jobs check --job-type TriggererJob --hostname "$HOSTNAME"` | 30 s / 10 s / 5 / 60 s |
| mlflow | `curl -f http://localhost:5000/health` | 30 s / 5 s / 5 / 30 s |
| prometheus | `wget -qO- http://localhost:9090/-/healthy` | 30 s / 5 s / 5 / 10 s |
| grafana | `curl -f http://localhost:3000/api/health` | 30 s / 5 s / 5 / 30 s |
| pushgateway / alertmanager | `/-/healthy` | 30 s / 5 s / 5 / 10 s |
| otel-collector | `wget -qO- http://localhost:13133` | 30 s / 5 s / 5 / 10 s |

## E.5 Dependency Conditions

`minio-init → minio(healthy)`; `airflow-init → postgres(healthy)`; `airflow-* → airflow-init(completed_successfully), postgres(healthy)`; `akl-api → postgres(healthy), qdrant(healthy), minio-init(completed_successfully)`; `mlflow → postgres(healthy), minio-init(completed)`; `grafana → prometheus(started)`.

## E.6 Resource Limits (prod overlay)

| Service | CPU | Memory |
|---|---|---|
| akl-api | 2 | 4 GB |
| airflow-scheduler | 2 | 4 GB |
| airflow-webserver | 1 | 1.5 GB |
| postgres | 1 | 2 GB |
| qdrant | 2 | 3 GB |
| minio | 1 | 1 GB |
| mlflow | 0.5 | 1 GB |
| prometheus | 0.5 | 1 GB |
| grafana | 0.5 | 512 MB |

---

# Appendix F — Metrics Catalog

All metrics defined in `akl/observability/metrics.py`. Types: C = Counter, G = Gauge, H = Histogram, S = Summary.

## F.1 API

| Metric | Type | Labels | Description |
|---|---|---|---|
| `akl_http_requests_total` | C | method, route, status | Requests |
| `akl_http_request_duration_seconds` | H | method, route | Latency |
| `akl_http_requests_inflight` | G | route | Concurrent requests |
| `akl_http_rate_limited_total` | C | route_class | 429s |
| `akl_auth_failures_total` | C | reason | 401/403 |
| `akl_uploads_total` | C | source_type, outcome | |
| `akl_upload_bytes_total` | C | source_type | |
| `akl_sse_streams_active` | G | — | |

## F.2 Retrieval

| Metric | Type | Labels | Description |
|---|---|---|---|
| `akl_search_requests_total` | C | mode, intent | |
| `akl_search_latency_seconds` | H | stage (query_processing, embed_query, dense, sparse, fusion, rerank, context, total) | |
| `akl_retrieval_candidates` | H | stage (dense, sparse, fused, final) | Candidate counts |
| `akl_rerank_confidence` | H | intent | Top rerank score |
| `akl_insufficient_evidence_total` | C | reason | Refusals |
| `akl_query_cache_hits_total` / `_misses_total` | C | — | |
| `akl_result_cache_hits_total` / `_misses_total` | C | — | |
| `akl_spell_corrections_total` | C | — | |
| `akl_soft_filter_fallback_total` | C | filter | Second pass invoked |
| `akl_retrieval_source_share` | H | backend (dense, sparse, both) | Share of final top-k by backend |
| `akl_qdrant_request_duration_seconds` | H | op | |
| `akl_bm25_search_duration_seconds` | H | — | |
| `akl_bm25_index_terms` | G | version | |
| `akl_bm25_index_docs` | G | version | |

## F.3 RAG

| Metric | Type | Labels | Description |
|---|---|---|---|
| `akl_chat_requests_total` | C | mode (generative, extractive), stream | |
| `akl_llm_latency_seconds` | H | phase (first_token, total) | |
| `akl_llm_tokens_total` | C | direction (input, output) | |
| `akl_llm_errors_total` | C | type | |
| `akl_llm_cost_usd_total` | C | model | |
| `akl_answer_citations` | H | — | Citations per answer |
| `akl_answer_flags_total` | C | flag | low_faithfulness, unsupported_token, llm_unavailable… |
| `akl_conversation_summaries_total` | C | — | |
| `akl_conversations_active` | G | — | |

## F.4 Ingestion

| Metric | Type | Labels |
|---|---|---|
| `akl_ingestion_documents_discovered_total` | C | connector |
| `akl_ingestion_documents_fetched_total` | C | connector |
| `akl_ingestion_fetch_errors_total` | C | connector, code |
| `akl_bronze_dedup_hits_total` | C | source_type |
| `akl_bronze_bytes_written_total` | C | source_type |
| `akl_ingestion_parsed_total` | C | parser, outcome |
| `akl_ingestion_parse_duration_seconds` | H | parser |
| `akl_quarantine_total` | C | code, stage, source_type |
| `akl_quarantine_open` | G | code |
| `akl_dedup_near_duplicates_total` | C | source_type |
| `akl_document_quality` | H | source_type |
| `akl_github_rate_limit_remaining` | G | repo |
| `akl_connector_state_age_seconds` | G | connector |

## F.5 Chunking

| Metric | Type | Labels |
|---|---|---|
| `akl_chunks_created_total` | C | status (unchanged, modified, moved, added, removed) |
| `akl_chunk_tokens` | H | chunk_type |
| `akl_chunk_quality` | H | chunk_type |
| `akl_chunking_duration_seconds` | H | — |
| `akl_documents_chunked_total` | C | source_type |
| `akl_chunks_low_quality_total` | C | source_type |

## F.6 Embedding and Vector

| Metric | Type | Labels |
|---|---|---|
| `akl_embedding_backlog_size` | G | embedding_version |
| `akl_embeddings_generated_total` | C | embedding_version, device |
| `akl_embedding_cache_hits_total` / `_misses_total` | C | embedding_version |
| `akl_embedding_batch_seconds` | H | device |
| `akl_embedding_throughput_cps` | G | shard |
| `akl_embedding_failures_total` | C | code |
| `akl_embedding_coverage_ratio` | G | embedding_version |
| `akl_qdrant_points` | G | collection |
| `akl_qdrant_points_upserted_total` / `_deleted_total` | C | collection |
| `akl_qdrant_gold_drift` | G | collection |
| `akl_qdrant_sync_duration_seconds` | H | — |
| `akl_bm25_build_seconds` | H | — |
| `akl_reindex_progress_ratio` | G | job_id |

## F.7 Lakehouse, Pipelines, Freshness

| Metric | Type | Labels |
|---|---|---|
| `akl_lakehouse_partition_files` | G | dataset, partition |
| `akl_lakehouse_bytes` | G | layer, dataset |
| `akl_lakehouse_rows` | G | dataset |
| `akl_compaction_files_merged_total` | C | dataset |
| `akl_compaction_bytes_before` / `_after` | G | dataset |
| `akl_retention_deleted_total` | C | dataset |
| `akl_dag_run_total` | C | dag_id, state |
| `akl_task_duration_seconds` | H | dag_id, task_id |
| `akl_task_retries_total` | C | dag_id, task_id |
| `akl_sla_miss_total` | C | dag_id |
| `akl_pipeline_last_success_timestamp` | G | dag_id |
| `akl_quality_gate_failures_total` | C | dag_id, gate |
| `akl_data_freshness_seconds` | G | source_type |
| `akl_gold_snapshot_age_seconds` | G | — |
| `akl_backup_bytes` | G | component |
| `akl_backup_last_success_timestamp` | G | component |

## F.8 Evaluation

| Metric | Type | Labels |
|---|---|---|
| `akl_eval_recall_at_k` | G | k, config |
| `akl_eval_mrr` | G | config |
| `akl_eval_ndcg_at_10` | G | config |
| `akl_eval_faithfulness` | G | prompt_version |
| `akl_eval_citation_coverage` | G | prompt_version |
| `akl_eval_refusal_precision` / `_recall` | G | — |
| `akl_eval_latency_p95_seconds` | G | config |

---

# Appendix G — Error Catalog

Format `AKL-E<domain><nn>` for errors, `AKL-W<domain><nn>` for warnings. Domains: 0 config, 1 auth, 2 lakehouse, 3 ingestion, 4 chunking, 5 embedding/vector, 6 RAG/API, 7 pipelines, 8 observability/ops, 9 governance.

| Code | HTTP | Retryable | Meaning | Recovery action |
|---|---|---|---|---|
| AKL-E0001 | 500 | no | Invalid/missing settings | Fix env; see startup log list |
| AKL-E0002 | 500 | no | Config YAML schema invalid | Fix file named in message |
| AKL-E1001 | 401 | no | Missing credentials | Provide JWT/API key |
| AKL-E1002 | 401 | no | Invalid/expired token | Re-authenticate |
| AKL-E1003 | 403 | no | Insufficient scope | Request role |
| AKL-E1004 | 403 | no | Not resource owner | — |
| AKL-E1005 | 401 | no | API key revoked/expired | Rotate key |
| AKL-E1006 | 429 | yes | Rate limited | Wait `Retry-After` |
| AKL-E2001 | 503 | yes | Object storage unreachable | Check MinIO health |
| AKL-E2002 | 500 | yes | Parquet write failed | Retry task |
| AKL-E2003 | 500 | no | Dataset/partition not found | Run upstream DAG |
| AKL-E2101 | 500 | no | Schema enforcement failed | Fix producer or bump schema version |
| AKL-E2102 | 500 | no | Compaction verification mismatch | Do not swap; inspect `_compacting` |
| AKL-E3001 | 413/422 | no | Document size out of bounds | Adjust limits or split doc |
| AKL-E3003 | 415 | no | Unsupported format | — |
| AKL-E3004 | 422 | no | Binary content in text source | — |
| AKL-E3005 | 422 | no | Insufficient text | — |
| AKL-W3006 | — | — | Low text density | Flag only |
| AKL-W3007 | — | — | Low language confidence | Flag only |
| AKL-E3008 | 422 | no | Secret-like content detected | Remove secret or override flag |
| AKL-E3009 | 422 | no | No security level assigned | Fix connector config |
| AKL-E3010 | — | no | Encrypted PDF | Provide decrypted file |
| AKL-E3011 | — | no | Corrupted PDF | Repair/re-export |
| AKL-E3012 | — | no | PDF exceeds page limit | Split |
| AKL-E3013 | — | no | Image-only PDF | OCR externally |
| AKL-W3014 | — | — | Heading depth > 8 | Flag |
| AKL-E3020 | — | yes | Source unreachable | Retry; check network/token |
| AKL-E3021 | — | yes | Partial fetch | Retry |
| AKL-E3022 | — | yes | Bronze write failed | Retry |
| AKL-E3023 | — | yes | Metadata DB unavailable | Retry |
| AKL-E3030 | — | no | Parser exception | Quarantine → retry after fix |
| AKL-E3031 | — | no | Parser timeout | Raise timeout or split |
| AKL-E3032 | — | no | Parser memory limit | Raise limit or split |
| AKL-E3040 | — | yes | GitHub rate limit exhausted | Wait reset; clone mode |
| AKL-E3041 | — | no | GitHub repo/branch not found | Fix config |
| AKL-E4001 | — | no | Chunk exceeds max tokens (invariant) | Bug: report |
| AKL-E4002 | — | no | Chunk config hash mismatch mid-run | Restart run |
| AKL-E4003 | — | no | Tokenizer load failed | Check models volume |
| AKL-E5001 | — | yes | Model download failed | Retry; check network |
| AKL-E5002 | — | yes | Embedding OOM | Auto batch halving |
| AKL-E5003 | — | no | Token overflow at embed | Chunker invariant broken |
| AKL-E5004 | — | no | NaN/Inf vector | Chunk flagged failed |
| AKL-E5005 | — | no | Model checksum mismatch | Verify `AKL_EMBED_MODEL_SHA256` |
| AKL-E5010 | 503 | no | Qdrant payload indexes missing | Run ensure step |
| AKL-E5011 | 503 | yes | Qdrant unavailable | Check service |
| AKL-E5012 | 500 | no | Collection dimension mismatch | Reindex into new collection |
| AKL-E5020 | — | yes | Qdrant/Gold drift after sync | RB-05 |
| AKL-E5021 | — | no | Alias points to unknown collection | Fix alias |
| AKL-E5030 | 503 | yes | BM25 index load failed | Rebuild |
| AKL-E6001 | 422 | no | Empty query | — |
| AKL-E6002 | 422 | no | Invalid filter | — |
| AKL-W6010 | 200 | — | Dense unavailable (degraded) | Check Qdrant |
| AKL-W6011 | 200 | — | Sparse unavailable (degraded) | Rebuild BM25 |
| AKL-E6012 | 503 | yes | All retrieval backends unavailable | — |
| AKL-W6013 | 200 | — | Reranker skipped | — |
| AKL-W6014 | 200 | — | Query embedding failed | — |
| AKL-W6020 | 200 | — | LLM unavailable → extractive | Check LLM endpoint |
| AKL-E6021 | 502 | yes | LLM error non-recoverable | — |
| AKL-E6030 | 404 | no | Conversation not found | — |
| AKL-E6031 | 403 | no | Conversation not owned | — |
| AKL-E6040 | 409 | no | Idempotency key reused with different body | Use new key |
| AKL-E7001 | — | no | Quality gate failed | Inspect gate metrics; fix; rerun |
| AKL-E7002 | — | yes | Downstream DAG trigger failed | Rerun trigger |
| AKL-E7003 | 502 | yes | Airflow API unreachable | Check webserver |
| AKL-E8001 | 503 | no | Embedding version mismatch API vs Qdrant | Promote/roll back version |
| AKL-E8002 | — | yes | Pushgateway unreachable | Metrics dropped; check |
| AKL-E9001 | 409 | no | Hard delete blocked by Object Lock | Wait retention |
| AKL-E9002 | 422 | no | Permission change invalid (lower than source floor) | — |
| AKL-E9003 | 500 | yes | Permission propagation partial | Rerun `permissions.apply` |

---

# Appendix H — Glossary

| Term | Definition |
|---|---|
| Bronze / Silver / Gold | Lakehouse layers: raw immutable / cleaned typed / AI-ready |
| Retrieval unit | A Gold chunk row eligible for indexing (current, non-duplicate, quality-passing) |
| `chunk_key` | Stable positional identity (document + heading path + ordinal) |
| `chunk_checksum` | Content identity of normalised chunk text |
| `lineage_id` | First `chunk_id` observed for a content lineage; used by citations across re-chunking |
| `embedding_version` | `model__version__dim` string tagging every vector |
| `gold_snapshot_id` | Run id of the last Gold refresh; used for cache keys and eval reproducibility |
| RRF | Reciprocal Rank Fusion |
| HNSW | Hierarchical Navigable Small World graph index |
| Derived state | Any store rebuildable from Gold (Qdrant, BM25, caches) |
| Quarantine | Holding area for inputs failing parse/validation |
| Soft / hard filter | Boost-first two-pass filter vs mandatory filter |
| Extractive mode | Answering by returning reranked passages without generation |
| Reconciliation | Diff-and-apply of Gold vs Qdrant |
| Blue/green reindex | Build new collection, swap alias, retire old |

---

*End of PRD — AKL-PRD-001 v1.0.0.*
