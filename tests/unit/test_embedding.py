"""Unit tests: embedding providers, cache codecs, Qdrant schema & payload (Milestones 21–24)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from akl.config import EmbeddingSettings, Settings
from akl.embedding.provider import (
    BgeOnnxProvider,
    EmbeddingModelError,
    HashEmbeddingProvider,
    build_provider,
    bytes_to_vector,
    cosine,
    unit_norm_ok,
    vector_to_bytes,
)
from akl.embedding.qdrant.reconciler import _payload
from akl.embedding.qdrant.schema import PAYLOAD_INDEXES, QdrantSchema, QdrantSchemaError
from akl.rag.query.filters import to_qdrant_filter
from akl.security.principal import Principal

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    return Settings.load(config_file=None, env_file=None)


def test_hash_provider_is_deterministic_normalised_and_similarity_aware() -> None:
    p = HashEmbeddingProvider()
    a = p.embed_documents(["MinIO stores raw bytes in the bronze layer"])[0]
    b = p.embed_documents(["Raw bytes are stored in MinIO bronze storage"])[0]
    c = p.embed_documents(["The cat sat on the mat"])[0]
    assert a.shape == (384,)
    assert a.dtype == np.float32
    assert unit_norm_ok(a)
    assert np.array_equal(a, p.embed_documents(["MinIO stores raw bytes in the bronze layer"])[0])
    assert cosine(a, b) > cosine(a, c)
    q = p.embed_query("where are raw bytes stored")
    assert cosine(q, a) > cosine(q, c)
    assert p.embedding_version == "hash-embed__1__384"


def test_vector_bytes_roundtrip_and_dim_check() -> None:
    vec = HashEmbeddingProvider(16).embed_documents(["abc"])[0]
    data = vector_to_bytes(vec)
    assert len(data) == 16 * 4
    assert np.allclose(bytes_to_vector(data, 16), vec)
    from akl.embedding.provider import EmbeddingError

    with pytest.raises(EmbeddingError):
        bytes_to_vector(data, 384)


def test_build_provider_and_version(settings: Settings, tmp_path: Path) -> None:
    hash_settings = settings.embedding.model_copy(update={"embed_provider": "hash"})
    assert isinstance(build_provider(hash_settings, tmp_path), HashEmbeddingProvider)
    assert settings.embedding.embedding_version == "bge-small-en-v1.5__1.5__384"
    bge = build_provider(settings.embedding, tmp_path, allow_download=False)
    assert isinstance(bge, BgeOnnxProvider)
    assert bge.embedding_version == settings.embedding.embedding_version
    with pytest.raises(EmbeddingModelError):  # no model files and downloads disabled
        bge.embed_documents(["x"])
    with pytest.raises(EmbeddingModelError):
        build_provider(EmbeddingSettings(embed_provider="nope"), tmp_path)


def test_payload_contract_epoch_and_untrusted() -> None:
    row = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "source_type": "html",
        "security_level": "public",
        "document_updated_at": datetime(2026, 9, 1, tzinfo=UTC),
        "text": "t",
        "heading_path": ["A"],
        "allowed_groups": [],
    }
    payload = _payload(row)
    assert payload["chunk_id"] == str(row["chunk_id"])
    assert payload["document_id"] == str(row["document_id"])
    assert payload["document_updated_at"] == int(row["document_updated_at"].timestamp())
    assert payload["untrusted"] is True
    assert payload["title"] is None  # every contract key present
    assert set(PAYLOAD_INDEXES) <= set(payload) | {"embedding_version"}


def test_qdrant_schema_in_memory_ensure_status_and_filtered_search(settings: Settings) -> None:
    client = QdrantClient(":memory:")
    schema = QdrantSchema(client, settings, dim=16, collection="kb_unit", manage_alias=False)
    st = schema.ensure()
    assert st.exists is True
    assert st.dim == 16
    assert st.points == 0
    # local in-memory Qdrant does not report payload indexes; the live component test asserts them
    assert set(st.missing_indexes) <= set(PAYLOAD_INDEXES)
    assert schema.ensure().exists is True  # idempotent
    with pytest.raises(QdrantSchemaError):
        QdrantSchema(client, settings, dim=32, collection="kb_unit", manage_alias=False).ensure()

    p = HashEmbeddingProvider(16)
    docs = {
        "a": ("internal", [], "how to configure the qdrant sync run"),
        "b": ("restricted", ["hr"], "employee salary bands and payroll"),
        "c": ("public", [], "welcome page of the wiki"),
    }
    points = []
    for name, (level, groups, text) in docs.items():
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
        payload = {
            "chunk_id": pid,
            "security_level": level,
            "allowed_groups": groups,
            "text": text,
            "source_type": "markdown",
            "document_updated_at": 1,
            "embedding_version": "hash-embed__1__16",
        }
        points.append(
            qm.PointStruct(id=pid, vector=p.embed_documents([text])[0].tolist(), payload=payload)
        )
    client.upsert("kb_unit", points=points, wait=True)
    assert schema.status().points == 3

    def search(principal: Principal, query: str) -> list[str]:
        vec = p.embed_query(query)
        res = client.query_points(
            "kb_unit",
            query=vec.tolist(),
            limit=5,
            query_filter=to_qdrant_filter(principal),
            with_payload=True,
        )
        return [str(pt.payload["text"]) for pt in res.points]  # type: ignore[index]

    reader = Principal("r", frozenset(), frozenset({"public", "internal"}))
    hr = Principal("h", frozenset({"hr"}), frozenset({"public", "internal", "restricted"}))
    eng_restricted = Principal(
        "e", frozenset({"eng"}), frozenset({"public", "internal", "restricted"})
    )
    assert "employee salary bands and payroll" not in search(reader, "payroll salary")
    assert "employee salary bands and payroll" in search(hr, "payroll salary")
    assert "employee salary bands and payroll" not in search(
        eng_restricted, "payroll salary"
    )  # level ok, group mismatch
    assert search(reader, "qdrant sync configuration")[0] == "how to configure the qdrant sync run"
