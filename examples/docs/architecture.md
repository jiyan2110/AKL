---
title: Architecture Overview
tags: [architecture, lakehouse]
---

# Architecture Overview

AKL treats organisational documents as a data engineering problem first and an AI problem
second. Sources are ingested into an immutable Bronze layer, refined into typed Silver datasets,
promoted into AI-ready Gold datasets, embedded, and indexed into a vector store that is treated
as a derived cache. If the vector database is deleted it is rebuilt from Gold.

## Layers

### Bronze

Raw bytes are stored under a content-addressed key: `bronze/raw/source_type=<t>/sha256=<hash>`.
Identical content under two URIs is stored once. The manifest dataset records every fetch.

### Silver

Parsers produce a unified document with a canonical text, character offsets for every block and
a heading tree. Chunks are derived from that structure with stable identities, so citations keep
resolving after small edits.

### Gold

Gold holds the retrieval units — the exact payload contract for Qdrant and BM25 — and every
embedding vector ever produced, keyed by the hash of the text that was embedded.

## Retrieval

Queries run dense (Qdrant) and sparse (BM25) retrieval in parallel, fuse results with Reciprocal
Rank Fusion, rerank with a cross-encoder and answer with citations. When evidence is weak the
system says so instead of guessing.

```python
def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores
```
