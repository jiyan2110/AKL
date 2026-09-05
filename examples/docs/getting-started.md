---
title: Getting Started with AKL
tags: [onboarding, docker]
---

# Getting Started

The Enterprise AI Knowledge Lakehouse (AKL) runs entirely on your laptop with Docker Compose.
This guide walks through the first run: starting the stack, seeding the example corpus and
asking your first question. Everything described here is free and open source.

## Prerequisites

- Docker Desktop or Docker Engine 24+
- Python 3.12 and `uv`
- 16 GB RAM recommended (8 GB minimum for the storage services alone)

## Start the stack

```bash
make up
make wait
uv run akl-cli config check
```

`make wait` blocks until PostgreSQL, MinIO and Qdrant report healthy. If a port is already in
use on your machine, override it in `.env` (for example `AKL_DEV_POSTGRES_PORT=55432`).

## Seed the example corpus

```bash
uv run akl-cli ingest run --connector markdown-examples
uv run akl-cli lakehouse silver-status
```

The connector stores every Markdown file content-addressed in the Bronze layer, records the
documents in PostgreSQL and parses them into the Silver layer. Running the same command a
second time performs no work: nothing changed, so nothing is fetched or parsed.

| Layer  | Storage        | What lives there                        |
|--------|----------------|-----------------------------------------|
| Bronze | MinIO          | Raw bytes plus an append-only manifest  |
| Silver | Parquet        | Parsed documents and chunks             |
| Gold   | Parquet        | Retrieval units and embeddings          |

## Troubleshooting

If `ingest run` reports quarantined items, inspect them with `uv run akl-cli ingest quarantine`.
Each item names the failing rule (for example `AKL-E3005` for documents with too little text).
