"""Unit tests: query processing (Milestone 25)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from akl.config import RetrievalSettings
from akl.rag.query import QueryProcessor
from akl.rag.query.entities import extract_entities
from akl.rag.query.filters import (
    MetadataFilters,
    allowed_by_filters,
    infer_soft_filters,
    to_qdrant_filter,
)
from akl.rag.query.intent import Intent, classify_intent
from akl.rag.query.normalize import EmptyQueryError, normalize_query
from akl.rag.query.spell import SpellCorrector
from akl.security.principal import Principal

pytestmark = pytest.mark.unit


def test_normalize_protects_code_like_tokens() -> None:
    n = normalize_query(
        "  How do I fix AKL-E5020 in akl/embedding/qdrant/reconciler.py  with AKL_QDRANT_URL and `make up`? \x00"
    )
    assert n.text.startswith("How do I fix AKL-E5020")
    assert "\x00" not in n.text
    assert {
        "AKL-E5020",
        "akl/embedding/qdrant/reconciler.py",
        "AKL_QDRANT_URL",
        "make up",
    } <= n.protected
    assert n.lowered == n.text.lower()
    with pytest.raises(EmptyQueryError):
        normalize_query("   ")


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("what port does MinIO use", Intent.FACTUAL_LOOKUP),
        ("How do I configure retries for the ingestion DAG?", Intent.HOW_TO),
        ("function that builds the BM25 index", Intent.CODE_SEARCH),
        ("AKL-E5020 drift error after sync", Intent.TROUBLESHOOTING),
        ("difference between Silver and Gold", Intent.COMPARISON),
        ("summarize the security chapter", Intent.SUMMARIZATION),
        ("thanks!", Intent.CHITCHAT),
    ],
)
def test_intent_rules(text: str, intent: Intent) -> None:
    assert classify_intent(text) == intent


def test_entities_and_exact_terms() -> None:
    e = extract_entities(
        'Why does AKL_CHUNK_MAX_TOKENS in configs/settings.yaml break "gold refresh" for org/akl-docs v1.2.3 and chunk_config_hash?',
        known_repos=["org/akl-docs"],
    )
    assert e.env_vars == ["AKL_CHUNK_MAX_TOKENS"]
    assert e.paths == ["configs/settings.yaml"]
    assert e.repos == ["org/akl-docs"]
    assert e.versions == ["v1.2.3"]
    assert "chunk_config_hash" in e.identifiers
    assert e.quoted == ["gold refresh"]
    assert "AKL_CHUNK_MAX_TOKENS" in e.exact_terms
    assert "gold refresh" in e.exact_terms


def test_soft_filter_inference() -> None:
    now = datetime(2026, 9, 5, tzinfo=UTC)
    e = extract_entities(
        "show me the latest pdf spec in python from repo org/docs", known_repos=["org/docs"]
    )
    soft = infer_soft_filters(
        "show me the latest pdf spec in python from repo org/docs", e, now=now
    )
    assert soft.repos == ["org/docs"]
    assert soft.source_types == ["pdf"]
    assert soft.code_languages == ["python"]
    assert soft.updated_after is not None
    assert (now - soft.updated_after).days == 90
    assert infer_soft_filters(
        "what is bronze", extract_entities("what is bronze"), now=now
    ).is_empty()


def test_spell_corrector_only_touches_oov_unprotected_tokens() -> None:
    sc = SpellCorrector.from_texts(
        ["the lakehouse stores parquet files in minio buckets and qdrant vectors"] * 3
    )
    fixed, corrections = sc.correct(
        "lakehuose parqet files with AKL_S3_BUCKET and reconciler", frozenset({"AKL_S3_BUCKET"})
    )
    assert corrections == {"lakehuose": "lakehouse", "parqet": "parquet"}
    assert "AKL_S3_BUCKET" in fixed
    assert "reconciler" in fixed  # OOV but no close vocab word → untouched
    assert sc.correct_token("minio") is None
    assert len(sc) == len(
        {
            w
            for w in "the lakehouse stores parquet files in minio buckets and qdrant vectors".split()
            if len(w) >= 3
        }
    )


def test_qdrant_filter_parity_with_in_memory_predicate() -> None:
    principal = Principal("u", frozenset({"eng"}), frozenset({"public", "internal"}))
    hard = MetadataFilters(source_types=["github"], updated_after=datetime(2026, 1, 1, tzinfo=UTC))
    f = to_qdrant_filter(principal, hard).model_dump(exclude_none=True)
    keys = [c["key"] for c in f["must"]]
    assert keys == ["security_level", "source_type", "document_updated_at"]
    assert f["min_should"]["min_count"] == 1
    rows = [
        {
            "security_level": "internal",
            "allowed_groups": [],
            "source_type": "github",
            "document_updated_at": datetime(2026, 5, 1, tzinfo=UTC),
        },
        {
            "security_level": "internal",
            "allowed_groups": ["hr"],
            "source_type": "github",
            "document_updated_at": datetime(2026, 5, 1, tzinfo=UTC),
        },
        {
            "security_level": "restricted",
            "allowed_groups": [],
            "source_type": "github",
            "document_updated_at": datetime(2026, 5, 1, tzinfo=UTC),
        },
        {
            "security_level": "internal",
            "allowed_groups": ["eng"],
            "source_type": "pdf",
            "document_updated_at": datetime(2026, 5, 1, tzinfo=UTC),
        },
        {
            "security_level": "internal",
            "allowed_groups": ["eng"],
            "source_type": "github",
            "document_updated_at": datetime(2025, 5, 1, tzinfo=UTC),
        },
    ]
    assert [allowed_by_filters(r, principal, hard) for r in rows] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_processor_dual_variants_and_hard_overrides_soft() -> None:
    sc = SpellCorrector.from_texts(["lakehouse ingestion pipeline"] * 3)
    qp = QueryProcessor(RetrievalSettings(), spell=sc, known_repos=["org/akl-docs"])
    q = qp.process(
        "latest lakehuose ingestion from repo org/akl-docs",
        Principal.dev(),
        filters=MetadataFilters(repos=["other/repo"]),
    )
    assert q.corrections == {"lakehuose": "lakehouse"}
    assert len(q.dense_variants) == 2
    assert q.sparse_text == "latest lakehouse ingestion from repo org/akl-docs"
    assert q.hard_filters.repos == ["other/repo"]
    assert q.soft_filters.repos == []  # user filter wins on that dimension
    assert q.soft_filters.updated_after is not None
    assert q.intent is Intent.FACTUAL_LOOKUP
    assert q.profile["top_k"] == 8
    single = QueryProcessor(RetrievalSettings(query_spell_dual=False), spell=sc).process(
        "lakehuose", Principal.dev()
    )
    assert single.dense_variants == ("lakehouse",)
    assert single.dense_text == "lakehouse"
