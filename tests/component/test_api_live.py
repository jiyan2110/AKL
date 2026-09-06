"""Component test: FastAPI gateway against the live stack (Milestones 31–36).

Builds a real RAGService (hash provider, private Qdrant collection, private BM25 version,
mock OpenAI-compatible LLM) into an AppState and drives the HTTP API with TestClient:
health, search, chat (generative, non-stream + SSE), conversations, upload (pipeline stubbed),
documents listing/detail/delete, jobs. Everything created is removed afterwards.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, text

from akl import __version__
from akl.api.deps import AppState, JobRegistry
from akl.api.main import create_app
from akl.api.middleware.ratelimit import TokenBucketLimiter
from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.models import (
    Conversation,
    Document,
    EmbeddingCache,
    EmbeddingJob,
    QuarantineItem,
    RetrievalTrace,
)
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.documents import DocumentRepository
from akl.db.session import Database, DatabaseUnavailableError
from akl.embedding.bm25.index import LATEST_KEY, Bm25Index
from akl.embedding.pipeline import EmbeddingPipeline
from akl.embedding.provider import HashEmbeddingProvider
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.ingestion.connectors.markdown import MarkdownConnector, MarkdownConnectorConfig
from akl.ingestion.service import IngestionService
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import Layer
from akl.rag.llm.provider import OpenAICompatProvider
from akl.rag.retrieval.rerank import LexicalReranker
from akl.rag.service import RAGService
from akl.security.auth import Authenticator

pytestmark = pytest.mark.component

DOC_A = """# Backup Runbook

## Nightly backups

Back up PostgreSQL every night with pg_dump and copy the archive to the backups prefix in MinIO.
Snapshot the Qdrant collection after each successful sync run and keep fourteen days of history.
Restore drills run monthly and must finish inside one hour for one hundred thousand chunks.
"""

DOC_B = """# Onboarding Guide

## First week

New engineers install Docker Desktop, clone the repository and run make up on their first morning.
The example corpus is ingested with make seed and questions are asked through the search endpoint.
Mentors review the pull request template and the conventional commit rules on Friday afternoon.
"""

CANNED = "PostgreSQL is backed up nightly with pg_dump and the archive goes to the MinIO backups prefix [1]. Qdrant is snapshotted after every sync [1]."


def _llm_transport() -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if body.get("stream"):
            parts = [CANNED[i : i + 9] for i in range(0, len(CANNED), 9)]
            data = (
                "".join(
                    f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n"
                    for p in parts
                )
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=data.encode(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": CANNED}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 40},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    try:
        settings = Settings.load()
        db = Database(settings)
        db.ping()
    except (AKLError, DatabaseUnavailableError) as exc:  # pragma: no cover
        pytest.skip(f"stack unavailable: {exc}")
    engine = DuckDBEngine(settings)
    tag = uuid.uuid4().hex[:8]
    ingest = IngestionService(settings, engine, db)
    chunking = ChunkingService(settings, engine, db)
    provider = HashEmbeddingProvider(settings.embedding.embed_dim)
    pipeline = EmbeddingPipeline(settings, engine, db, provider=provider)
    try:
        ingest.io.ensure_bucket()
        client = make_client(settings)
        client.get_collections()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MinIO/Qdrant unavailable: {exc}")
    collection = f"kb_ctest_{tag}"
    reconciler = QdrantReconciler(
        client, settings, engine, pipeline.gold, collection=collection, manage_alias=False
    )
    root = tmp_path / "docs"
    root.mkdir()
    cfg = MarkdownConnectorConfig(
        id=f"ctest-{tag}",
        type="markdown",
        root_path=root,
        uri_base=f"https://ctest.example.com/{tag}",
        fetch_concurrency=2,
    )
    latest_before = (
        ingest.io.get_object(LATEST_KEY) if ingest.io.object_exists(LATEST_KEY) else None
    )
    run_ids: list[str] = []
    shas: list[str] = []

    # --- corpus: ingest → chunk → embed → private qdrant → private bm25 -------------------------------
    (root / "backup.md").write_text(DOC_A, encoding="utf-8")
    (root / "onboarding.md").write_text(DOC_B, encoding="utf-8")
    r1 = f"ctest-{cfg.id}-r1"
    run_ids.append(r1)
    ingest.run_connector(connector=MarkdownConnector(cfg), run_id=r1)
    ingest.parse_backlog(run_id=r1)
    with db.session() as s:
        repo = DocumentRepository(s)
        doc_ids = [
            str(repo.get_by_uri(f"{cfg.uri_base}/{n}.md").document_id)
            for n in ("backup", "onboarding")
        ]  # type: ignore[union-attr]
    chunking.run(run_id=r1, document_ids=[uuid.UUID(d) for d in doc_ids])
    pipeline.run(run_id=r1, document_ids=doc_ids)
    with db.session() as s:
        for d in doc_ids:
            shas.extend(
                c.embedded_text_sha256
                for c in ChunkRepository(s).current_for_document(uuid.UUID(d))
                if c.embedded_text_sha256
            )
    reconciler.sync(run_id=r1)
    units = pipeline.gold.active_units(
        where=f"document_id IN ('{doc_ids[0]}', '{doc_ids[1]}')"
    ).to_pylist()
    Bm25Index.build(units, version=r1).save(ingest.io)

    llm_settings = settings.llm.model_copy(
        update={
            "llm_provider": "openai_compat",
            "llm_model": "fake-llm",
            "llm_base_url": "http://llm.local/v1",
        }
    )
    llm = OpenAICompatProvider(llm_settings, transport=_llm_transport())
    principal_id = f"ctest-{tag}"
    api_settings = settings.model_copy(
        update={
            "api": settings.api.model_copy(
                update={
                    "auth_disabled": False,
                    "jwt_secret": SecretStr("ctest-jwt-secret-" + "x" * 32),
                }
            )
        }
    )
    rag = RAGService(
        api_settings,
        engine,
        db,
        provider=provider,
        reranker=LexicalReranker(),
        bm25=Bm25Index.load(ingest.io),
        qdrant_collection=collection,
        manage_alias=False,
        qdrant_use_alias=False,
        allow_download=False,
        llm=llm,
    )
    state = AppState(
        settings=api_settings,
        engine=engine,
        db=db,
        rag=rag,
        authenticator=Authenticator(api_settings, db),
        limiter=TokenBucketLimiter({}, default_rpm=10_000),
        jobs=JobRegistry(),
        version=__version__,
    )

    # never run the real post-ingest pipeline (it would embed with the production model and re-sync the alias)
    pipeline_calls: list[dict[str, Any]] = []

    def fake_pipeline(settings_: Settings, db_: Database, **kw: Any) -> Any:
        from akl.pipelines.local import LocalPipelineReport

        pipeline_calls.append(kw)
        return LocalPipelineReport(
            run_id=kw.get("run_id") or "x",
            parsed=len(kw.get("document_ids") or []),
            stages=["parse", "chunk"],
        )

    monkeypatch.setattr("akl.api.routers.documents.run_post_ingest", fake_pipeline)

    token = state.authenticator.mint_token(
        principal_id,
        groups=["eng"],
        security_levels=["public", "internal", "restricted"],
        roles=["curator"],
    )
    ctx: dict[str, Any] = {
        "state": state,
        "doc_ids": doc_ids,
        "tag": tag,
        "pipeline_calls": pipeline_calls,
        "principal_id": principal_id,
    }
    with TestClient(
        create_app(api_settings, state=state),
        headers={"Authorization": f"Bearer {token}"},
    ) as c:
        ctx["client"] = c
        yield ctx
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
    finally:
        for layer, dataset in (
            (Layer.SILVER, "documents"),
            (Layer.SILVER, "chunks"),
            (Layer.GOLD, "retrieval_units"),
            (Layer.GOLD, "chunk_embeddings"),
            (Layer.QUARANTINE, "reasons"),
            (Layer.BRONZE, "manifest"),
        ):
            keys = [
                f.key
                for f in ingest.io.list_files(layer, dataset)
                if any(r in f.key for r in run_ids) or "api-" in f.key and tag in f.key
            ]
            if keys:
                ingest.io.delete_keys(keys)
        for rid in run_ids:
            for name in ("index.jsonl.gz", "meta.json"):
                key = f"gold/indexes/bm25/version={rid}/{name}"
                if ingest.io.object_exists(key):
                    ingest.io.delete_keys([key])
        if latest_before is not None:
            ingest.io.put_object(LATEST_KEY, latest_before, content_type="text/plain")
        elif ingest.io.object_exists(LATEST_KEY):
            ingest.io.delete_keys([LATEST_KEY])
        with db.session() as s:
            s.execute(delete(QuarantineItem).where(QuarantineItem.run_id.in_(run_ids)))
            s.execute(delete(EmbeddingJob).where(EmbeddingJob.run_id.in_(run_ids)))
            s.execute(delete(RetrievalTrace).where(RetrievalTrace.principal_id == principal_id))
            s.execute(delete(Conversation).where(Conversation.principal_id == principal_id))
            if shas:
                s.execute(
                    delete(EmbeddingCache).where(
                        EmbeddingCache.embedded_text_sha256.in_(shas),
                        EmbeddingCache.model_id == provider.model_id,
                    )
                )
            s.execute(
                delete(Document).where(
                    Document.connector_id.like("ctest-%")
                    | (Document.canonical_source_uri.like(f"upload://{principal_id}/%"))
                )
            )
            s.execute(text("DELETE FROM connector_state WHERE connector_id LIKE 'ctest-%'"))
        engine.close()
        db.dispose()


def test_api_end_to_end(api: dict[str, Any]) -> None:
    c: TestClient = api["client"]
    doc_ids: list[str] = api["doc_ids"]

    # --- auth: no token → 401; health is public -------------------------------------------------------------
    denied = c.post(
        "/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer not-a-valid-token"}
    )
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "AKL-E1002"

    # --- health ---------------------------------------------------------------------------------
    ready = c.get("/v1/health/ready")
    assert ready.status_code == 200, ready.text
    names = {d["name"]: d for d in ready.json()["dependencies"]}
    assert names["postgres"]["ok"] and names["qdrant"]["ok"] and names["bm25"]["ok"]  # noqa: PT018
    assert ready.json()["embedding_version"] == "hash-embed__1__384"
    assert "akl_http_requests_total" in c.get("/metrics").text

    # --- search (hybrid) ----------------------------------------------------------------------------
    r = c.post("/v1/search", json={"query": "how are nightly postgres backups taken", "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sufficient"] is True
    assert body["results"][0]["document_id"] == doc_ids[0]
    assert body["results"][0]["scores"]["sparse"] is not None
    assert body["results"][0]["scores"]["dense"] is not None
    only_b = c.post(
        "/v1/search", json={"query": "nightly backups", "filters": {"document_id": [doc_ids[1]]}}
    ).json()
    assert all(x["document_id"] == doc_ids[1] for x in only_b["results"])

    # --- chat: generative, non-stream, memory --------------------------------------------------------
    chat = c.post(
        "/v1/chat", json={"query": "How are nightly postgres backups taken?", "include_trace": True}
    )
    assert chat.status_code == 200, chat.text
    answer = chat.json()
    assert answer["mode"] == "generative"
    assert answer["llm"]["model"] == "fake-llm"
    assert answer["answer"].startswith("PostgreSQL is backed up nightly")
    assert answer["citations"]
    assert answer["citations"][0]["document_id"] == doc_ids[0]
    assert "no_citations" not in answer["flags"]
    conv_id = answer["conversation_id"]
    assert uuid.UUID(conv_id)
    follow = c.post(
        "/v1/chat", json={"query": "How often is it snapshotted?", "conversation_id": conv_id}
    )
    assert follow.status_code == 200
    assert follow.json()["conversation_id"] == conv_id
    assert follow.json()["retrieval"]["rewritten_query"]  # anaphora resolved via memory
    conv = c.get(f"/v1/conversations/{conv_id}").json()
    assert conv["turn_count"] == 4
    assert [t["role"] for t in conv["turns"]] == ["user", "assistant", "user", "assistant"]
    assert c.get(f"/v1/conversations/{uuid.uuid4()}").status_code == 404

    # --- admin: permissions, API keys, audit, GDPR, PII (Milestones 49-52) --------------------------
    principal_id = api["principal_id"]
    perm = c.patch(
        f"/v1/admin/documents/{doc_ids[0]}/permissions",
        json={"security_level": "restricted", "allowed_groups": ["ops"]},
    )
    assert perm.status_code == 200, perm.text
    assert perm.json() == {
        "document_id": doc_ids[0],
        "security_level": "restricted",
        "allowed_groups": ["ops"],
    }
    restricted_visible = c.get(f"/v1/documents/{doc_ids[0]}")
    assert (
        restricted_visible.status_code == 404
    )  # our token's groups={"eng"} don't intersect allowed_groups=["ops"]
    c.patch(
        f"/v1/admin/documents/{doc_ids[0]}/permissions",
        json={"security_level": "internal", "allowed_groups": []},
    )  # restore

    created = c.post("/v1/admin/api-keys", json={"name": "ci-probe", "roles": ["reader"]})
    assert created.status_code == 201, created.text
    key_id = created.json()["key_id"]
    assert created.json()["token"].startswith("akl_")
    listed = c.get("/v1/admin/api-keys").json()
    assert any(k["key_id"] == key_id for k in listed)
    revoke = c.delete(f"/v1/admin/api-keys/{key_id}")
    assert revoke.status_code == 204
    assert (
        next(k for k in c.get("/v1/admin/api-keys").json() if k["key_id"] == key_id)["revoked_at"]
        is not None
    )
    bad_key = c.post(
        "/v1/search", json={"query": "x"}, headers={"X-API-Key": "akl_deadbeef_not-a-real-secret"}
    )
    assert (
        bad_key.status_code == 401
    )  # a real database is present here, so this is a genuine "wrong key" rejection
    assert bad_key.json()["error"]["code"] == "AKL-E1005"

    audit_rows = c.get("/v1/admin/audit", params={"principal_id": principal_id, "limit": 50}).json()
    actions = {r["action"] for r in audit_rows}
    assert {"document.permissions.update", "api_key.create", "api_key.revoke"} <= actions

    export = c.get(f"/v1/admin/gdpr/principals/{principal_id}/export")
    assert export.status_code == 200, export.text
    assert export.json()["principal_id"] == principal_id
    assert len(export.json()["conversations"]) >= 1
    erase = c.delete(f"/v1/admin/gdpr/principals/{principal_id}")
    assert erase.status_code == 200
    assert erase.json()["conversations_deleted"] >= 1
    assert c.get(f"/v1/conversations/{conv_id}").status_code == 404  # actually erased
    forbidden_gdpr = c.delete("/v1/admin/gdpr/principals/someone-else")
    assert forbidden_gdpr.status_code == 403

    pii_doc = c.post(
        "/v1/documents",
        files=[
            (
                "files",
                (
                    "contact.md",
                    b"# Contact\n\nReach the on-call engineer at oncall@example.com.\n",
                    "text/markdown",
                ),
            )
        ],
        data={"security_level": "internal"},
    )
    assert pii_doc.status_code == 202, pii_doc.text
    pii_document_id = pii_doc.json()["items"][0]["document_id"]
    # PII scanning runs inside the real parse_backlog task (stubbed out by `fake_pipeline` in this
    # test's fixture, see test_pipeline_tasks_live.py for coverage against the real task entrypoint);
    # here we only confirm the lineage endpoint added in this batch is reachable for the new document.
    trace = c.get(f"/v1/admin/lineage/documents/{pii_document_id}")
    assert trace.status_code == 200, trace.text
    assert trace.json()["document_id"] == pii_document_id

    # --- hard delete: purges the raw Bronze object in addition to everything soft delete does ------------------
    bronze_key = trace.json()["versions"][0]["bronze_object_key"]
    io = api["state"].rag.io
    assert io.object_exists(bronze_key)
    no_confirm = c.delete(f"/v1/documents/{pii_document_id}", params={"mode": "hard"})
    assert no_confirm.status_code == 400
    assert no_confirm.json()["error"]["code"] == "AKL-E3060"
    hard = c.delete(
        f"/v1/documents/{pii_document_id}",
        params={"mode": "hard"},
        headers={"X-Confirm": "hard-delete"},
    )
    assert hard.status_code == 200, hard.text
    assert hard.json()["mode"] == "hard"
    assert not io.object_exists(bronze_key)  # raw bytes actually purged, unlike soft delete
    audit_hard = c.get(
        "/v1/admin/audit", params={"action": "document.delete.hard", "limit": 5}
    ).json()
    assert any(r["resource_id"] == pii_document_id for r in audit_hard)

    # --- chat: SSE stream ----------------------------------------------------------------------------
    with c.stream(
        "POST", "/v1/chat", json={"query": "how are nightly postgres backups taken", "stream": True}
    ) as s:
        assert s.status_code == 200
        raw = "".join(s.iter_text())
    kinds = [blk.split("\n", 1)[0].removeprefix("event: ") for blk in raw.strip().split("\n\n")]
    assert kinds[0] == "meta" and kinds[-2] == "citations" and kinds[-1] == "done"  # noqa: PT018
    tokens = [
        json.loads(blk.split("\n", 1)[1].removeprefix("data: "))["text"]
        for blk in raw.strip().split("\n\n")
        if blk.startswith("event: token")
    ]
    assert "".join(tokens).startswith("PostgreSQL is backed up nightly")
    assert not any(t.endswith("[1") for t in tokens)  # unfinished markers are buffered

    # --- extractive mode on request ------------------------------------------------------------------------
    ext = c.post(
        "/v1/chat", json={"query": "how are nightly postgres backups taken", "mode": "extractive"}
    ).json()
    assert ext["mode"] == "extractive" and ext["citations"]  # noqa: PT018

    # --- upload (pipeline stubbed) → documents → detail → delete ---------------------------------------------
    notes = f"# Notes\n\nSome uploaded notes about restore drills ({api['tag']}).\n".encode()
    up = c.post(
        "/v1/documents",
        files=[("files", ("notes.md", notes, "text/markdown"))],
        data={"security_level": "internal"},
    )
    assert up.status_code == 202, up.text
    item = up.json()["items"][0]
    assert item["status"] == "bronze" and item["deduplicated"] is False  # noqa: PT018
    assert api["pipeline_calls"]
    assert str(api["pipeline_calls"][-1]["document_ids"][0]) == item["document_id"]
    job = c.get(up.json()["status_url"].replace("http://testserver", ""))
    assert job.status_code == 200
    assert job.json()["state"] == "succeeded"
    again = c.post("/v1/documents", files=[("files", ("notes.md", notes, "text/markdown"))])
    assert again.json()["items"][0]["deduplicated"] is True
    bad = c.post("/v1/documents", files=[("files", ("archive.zip", b"PK", "application/zip"))])
    assert bad.status_code == 415
    listing = c.get("/v1/documents", params={"q": "notes"}).json()
    assert any(d["document_id"] == item["document_id"] for d in listing["items"])
    detail = c.get(f"/v1/documents/{item['document_id']}").json()
    assert detail["status"] == "bronze" and detail["versions"]  # noqa: PT018
    chunks = c.get(f"/v1/documents/{doc_ids[0]}/chunks").json()
    assert chunks["total_estimate"] >= 1
    deleted = c.delete(f"/v1/documents/{doc_ids[1]}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    after = c.post(
        "/v1/search", json={"query": "engineers first morning docker desktop", "k": 5}
    ).json()
    assert all(x["document_id"] != doc_ids[1] for x in after["results"])
    assert c.get(f"/v1/documents/{doc_ids[1]}").json()["status"] == "deleted"
    assert c.get("/v1/documents/not-a-uuid").status_code == 404
