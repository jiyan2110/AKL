"""Metadata filters (PRD §6.2.5) and the security predicate compiled to Qdrant filters (ADR-009)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client.http import models as qm

from akl.rag.query.entities import Entities
from akl.security.principal import Principal

_TEMPORAL = re.compile(
    r"\b(latest|recent(ly)?|newest|current|as of|this (month|quarter|year))\b", re.I
)
_LANGUAGE_HINT = re.compile(
    r"\bin (python|typescript|javascript|go|java|rust|sql|yaml|bash|shell)\b", re.I
)
_SOURCE_HINT = re.compile(r"\b(pdf|whitepaper|handbook|spec(ification)?)\b", re.I)
_REPO_HINT = re.compile(
    r"\b(in|from|inside) (repo|repository) ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I
)


@dataclass
class MetadataFilters:
    """Hard filters are mandatory; soft filters are applied as a first pass then relaxed (PRD §6.2.5)."""

    source_types: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    chunk_types: list[str] = field(default_factory=list)
    code_languages: list[str] = field(default_factory=list)
    updated_after: datetime | None = None
    document_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.source_types
            or self.repos
            or self.chunk_types
            or self.code_languages
            or self.updated_after
            or self.document_ids
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: v for k, v in self.__dict__.items() if v}
        if self.updated_after:
            out["updated_after"] = self.updated_after.isoformat()
        return out


def infer_soft_filters(
    text: str, entities: Entities, *, now: datetime | None = None
) -> MetadataFilters:
    soft = MetadataFilters()
    m = _REPO_HINT.search(text)
    if m:
        soft.repos.append(m.group(3))
    elif entities.repos:
        soft.repos.extend(entities.repos)
    if _SOURCE_HINT.search(text):
        soft.source_types.append("pdf")
    lang = _LANGUAGE_HINT.search(text)
    if lang:
        word = lang.group(1).lower()
        soft.code_languages.append({"shell": "bash"}.get(word, word))
    if _TEMPORAL.search(text):
        soft.updated_after = (now or datetime.now(UTC)) - timedelta(days=90)
    return soft


def security_conditions(principal: Principal) -> tuple[list[qm.Condition], list[qm.Condition]]:
    """``(must, should)`` implementing PRD §9.3 — level ∈ allowed ∧ (groups ∩ ≠ ∅ ∨ groups empty)."""
    must: list[qm.Condition] = [
        qm.FieldCondition(
            key="security_level", match=qm.MatchAny(any=sorted(principal.security_levels))
        )
    ]
    should: list[qm.Condition] = [
        qm.IsEmptyCondition(is_empty=qm.PayloadField(key="allowed_groups"))
    ]
    if principal.groups:
        should.append(
            qm.FieldCondition(key="allowed_groups", match=qm.MatchAny(any=sorted(principal.groups)))
        )
    return must, should


def to_qdrant_filter(
    principal: Principal, hard: MetadataFilters | None = None, soft: MetadataFilters | None = None
) -> qm.Filter:
    must, should = security_conditions(principal)
    for filters in (hard, soft):
        if filters is None:
            continue
        if filters.source_types:
            must.append(
                qm.FieldCondition(key="source_type", match=qm.MatchAny(any=filters.source_types))
            )
        if filters.repos:
            must.append(qm.FieldCondition(key="repo", match=qm.MatchAny(any=filters.repos)))
        if filters.chunk_types:
            must.append(
                qm.FieldCondition(key="chunk_type", match=qm.MatchAny(any=filters.chunk_types))
            )
        if filters.code_languages:
            must.append(
                qm.FieldCondition(
                    key="code_language", match=qm.MatchAny(any=filters.code_languages)
                )
            )
        if filters.document_ids:
            must.append(
                qm.FieldCondition(key="document_id", match=qm.MatchAny(any=filters.document_ids))
            )
        if filters.updated_after:
            must.append(
                qm.FieldCondition(
                    key="document_updated_at",
                    range=qm.Range(gte=int(filters.updated_after.timestamp())),
                )
            )
    return qm.Filter(
        must=must, should=should, min_should=qm.MinShould(conditions=should, min_count=1)
    )


def allowed_by_filters(
    row: dict[str, Any],
    principal: Principal,
    hard: MetadataFilters | None = None,
    soft: MetadataFilters | None = None,
) -> bool:
    """In-memory equivalent of :func:`to_qdrant_filter` (used by BM25 and by tests for parity)."""
    if not principal.can_read(str(row.get("security_level")), row.get("allowed_groups")):
        return False
    for filters in (hard, soft):
        if filters is None:
            continue
        if filters.source_types and row.get("source_type") not in filters.source_types:
            return False
        if filters.repos and row.get("repo") not in filters.repos:
            return False
        if filters.chunk_types and row.get("chunk_type") not in filters.chunk_types:
            return False
        if filters.code_languages and row.get("code_language") not in filters.code_languages:
            return False
        if filters.document_ids and str(row.get("document_id")) not in filters.document_ids:
            return False
        if filters.updated_after:
            ts = row.get("document_updated_at")
            epoch = ts.timestamp() if isinstance(ts, datetime) else float(ts or 0)
            if epoch < filters.updated_after.timestamp():
                return False
    return True
