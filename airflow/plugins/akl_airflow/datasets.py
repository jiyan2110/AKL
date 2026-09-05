"""Dataset URIs for data-aware scheduling (PRD §7.1)."""

from __future__ import annotations

from airflow.datasets import Dataset

SILVER_DOCUMENTS = Dataset("akl://silver/documents")
SILVER_CHUNKS = Dataset("akl://silver/chunks")
GOLD_EMBEDDINGS = Dataset("akl://gold/chunk_embeddings")
