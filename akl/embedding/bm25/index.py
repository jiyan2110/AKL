"""BM25 index build/serialise/load/search (PRD §6.3.2, ADR-008).

Artefact layout (MinIO): ``gold/indexes/bm25/version=<gold_snapshot_id>/index.jsonl.gz`` +
``meta.json``; ``gold/indexes/bm25/LATEST`` holds the current version id. The corpus is stored as
JSON (chunk payload + tokens) — no pickle — and ``BM25Okapi`` is rebuilt in memory on load.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from akl.embedding.bm25.tokenizer import tokenize
from akl.errors import AKLError
from akl.lakehouse.io import LakehouseIO

INDEX_PREFIX = "gold/indexes/bm25"
LATEST_KEY = f"{INDEX_PREFIX}/LATEST"
PAYLOAD_KEYS: tuple[str, ...] = (
    "chunk_id",
    "lineage_id",
    "document_id",
    "document_version_id",
    "chunk_index",
    "source_type",
    "canonical_source_uri",
    "source_uri",
    "title",
    "heading_path",
    "heading_breadcrumb",
    "chunk_type",
    "code_language",
    "text",
    "context_prefix",
    "token_count",
    "page_start",
    "page_end",
    "line_start",
    "line_end",
    "security_level",
    "allowed_groups",
    "repo",
    "branch",
    "path",
    "document_updated_at",
    "quality_score",
    "quality_flags",
    "language",
    "gold_snapshot_id",
    "embedded_text_sha256",
)


class _PositiveIdfBM25(BM25Okapi):  # type: ignore[misc]
    """BM25 using Lucene/Robertson positive IDF."""

    def _calc_idf(self, nd: dict[str, int]) -> None:
        self.idf = {
            word: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for word, freq in nd.items()
        }
        self.average_idf = sum(self.idf.values()) / max(1, len(self.idf))


class Bm25IndexError(AKLError):
    """Index missing or unloadable (AKL-E5030)."""

    code = "AKL-E5030"
    http_status = 503
    retryable = True


@dataclass(frozen=True)
class SparseHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return int(value.timestamp())
    if hasattr(value, "hex") and not isinstance(value, str | bytes):  # uuid
        return str(value)
    return value


class Bm25Index:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        tokens: list[list[str]],
        *,
        k1: float,
        b: float,
        version: str,
        built_at: str | None = None,
    ) -> None:
        if len(payloads) != len(tokens):
            raise Bm25IndexError("payload/token length mismatch", retryable=False)
        self.payloads = payloads
        self.tokens = tokens
        self.k1, self.b = k1, b
        self.version = version
        self.built_at = built_at or datetime.now(UTC).isoformat()
        self._bm25 = _PositiveIdfBM25(tokens if tokens else [["__empty__"]], k1=k1, b=b)
        self._exact: dict[str, set[int]] = {}
        for i, toks in enumerate(tokens):
            for t in set(toks):
                if any(ch in t for ch in "_./-") or t.startswith("akl-"):
                    self._exact.setdefault(t, set()).add(i)
        self.excluded: set[str] = (
            set()
        )  # chunk_ids removed since build (deletions before next rebuild)

    # -- build -------------------------------------------------------------------------
    @classmethod
    def build(
        cls, rows: Sequence[dict[str, Any]], *, version: str, k1: float = 1.5, b: float = 0.75
    ) -> Bm25Index:
        payloads = [{k: _jsonable(r.get(k)) for k in PAYLOAD_KEYS} for r in rows]
        tokens = [
            tokenize(
                f"{r.get('context_prefix') or ''}\n{r.get('text') or ''}", keep_stopwords=False
            )
            for r in rows
        ]
        return cls(payloads, tokens, k1=k1, b=b, version=version)

    @property
    def size(self) -> int:
        return len(self.payloads)

    @property
    def terms(self) -> int:
        return len(getattr(self._bm25, "idf", {}))

    # -- search --------------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
        exact_terms: Sequence[str] = (),
        oversample: int = 4,
    ) -> list[SparseHit]:
        if self.size == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float64)
        exact_mask: set[int] | None = None
        for term in exact_terms:
            hits = self._exact.get(term.lower())
            if hits:
                exact_mask = set(hits) if exact_mask is None else exact_mask | set(hits)
        if exact_mask:
            boost = np.zeros_like(scores)
            boost[list(exact_mask)] = scores.max() if scores.max() > 0 else 1.0
            scores = scores + boost  # exact identifier matches rank first (PRD §6.3.2 must-terms)
        order = np.argsort(-scores)[: max(k * oversample, k)]
        out: list[SparseHit] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            payload = self.payloads[int(idx)]
            if payload["chunk_id"] in self.excluded:
                continue
            if allowed is not None and not allowed(payload):
                continue
            out.append(SparseHit(str(payload["chunk_id"]), float(scores[idx]), payload))
            if len(out) >= k:
                break
        return out

    def exclude(self, chunk_ids: Sequence[str]) -> None:
        self.excluded.update(chunk_ids)

    # -- persistence --------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "documents": self.size,
            "terms": self.terms,
            "k1": self.k1,
            "b": self.b,
        }

    def save(self, io: LakehouseIO) -> str:
        prefix = f"{INDEX_PREFIX}/version={self.version}"
        lines = "\n".join(
            json.dumps({"p": p, "t": t}, ensure_ascii=False)
            for p, t in zip(self.payloads, self.tokens, strict=True)
        )
        io.put_object(
            f"{prefix}/index.jsonl.gz",
            gzip.compress(lines.encode("utf-8")),
            content_type="application/gzip",
        )
        io.put_object(
            f"{prefix}/meta.json",
            json.dumps(self.meta()).encode("utf-8"),
            content_type="application/json",
        )
        io.put_object(LATEST_KEY, self.version.encode("utf-8"), content_type="text/plain")
        return prefix

    @classmethod
    def load(
        cls, io: LakehouseIO, *, version: str | None = None, k1: float = 1.5, b: float = 0.75
    ) -> Bm25Index:
        start = time.perf_counter()
        try:
            ver = version or cls.latest_version(io)
            if not ver:
                raise Bm25IndexError(
                    "no BM25 index versions found; run `akl-cli qdrant sync` or `akl-cli bm25 build`"
                )
            prefix = f"{INDEX_PREFIX}/version={ver}"
            meta = json.loads(io.get_object(f"{prefix}/meta.json"))
            raw = gzip.decompress(io.get_object(f"{prefix}/index.jsonl.gz")).decode("utf-8")
        except Bm25IndexError:
            raise
        except AKLError as exc:
            raise Bm25IndexError(
                "BM25 index unreadable; run `akl-cli qdrant sync` or `akl-cli bm25 build`",
                details={"version": version, "error": exc.message, **exc.details},
            ) from exc
        except (OSError, ValueError, KeyError) as exc:
            raise Bm25IndexError(
                "BM25 artefact is corrupt", details={"error": f"{type(exc).__name__}: {exc}"}
            ) from exc
        payloads: list[dict[str, Any]] = []
        tokens: list[list[str]] = []
        for line in raw.splitlines():
            if line:
                rec = json.loads(line)
                payloads.append(rec["p"])
                tokens.append(rec["t"])
        index = cls(
            payloads,
            tokens,
            k1=meta.get("k1", k1),
            b=meta.get("b", b),
            version=ver,
            built_at=meta.get("built_at"),
        )
        index.load_seconds = time.perf_counter() - start  # type: ignore[attr-defined]
        return index

    @staticmethod
    def latest_version(io: LakehouseIO) -> str | None:
        """Version from the LATEST pointer, else the newest ``version=`` prefix that has a meta.json."""
        if io.object_exists(LATEST_KEY):
            ver = io.get_object(LATEST_KEY).decode("utf-8").strip()
            if ver and io.object_exists(f"{INDEX_PREFIX}/version={ver}/meta.json"):
                return ver
        versions = sorted(
            key.split("version=", 1)[1].split("/", 1)[0]
            for key in io.list_keys(f"{INDEX_PREFIX}/version=")
            if key.endswith("/meta.json")
        )
        return versions[-1] if versions else None
