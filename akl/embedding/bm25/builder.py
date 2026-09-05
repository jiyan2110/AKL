"""Build the BM25 artefact from Gold active units (PRD §7.6 rebuild_bm25_index task)."""

from __future__ import annotations

from dataclasses import dataclass

from akl.config import Settings
from akl.embedding.bm25.index import Bm25Index
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO


@dataclass(frozen=True)
class Bm25BuildReport:
    version: str
    documents: int
    terms: int
    prefix: str


def build_bm25(
    settings: Settings, io: LakehouseIO, gold: GoldStore, *, version: str
) -> Bm25BuildReport:
    """Snapshot all active Gold units into a new BM25 index version and mark it LATEST."""
    rows = gold.active_units().to_pylist()
    index = Bm25Index.build(
        rows, version=version, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b
    )
    prefix = index.save(io)
    return Bm25BuildReport(version=version, documents=index.size, terms=index.terms, prefix=prefix)
