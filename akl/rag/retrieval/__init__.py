"""Hybrid retrieval (PRD §6.3): dense (Qdrant) + sparse (BM25) → RRF fusion → cross-encoder rerank."""

from akl.rag.retrieval.engine import HybridRetriever, RetrievalResult
from akl.rag.retrieval.models import Candidate

__all__ = ["Candidate", "HybridRetriever", "RetrievalResult"]
