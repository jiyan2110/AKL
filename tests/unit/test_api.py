"""Unit tests: FastAPI gateway with a fake RAG service (Milestones 31–36)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_retrieval import rows

from akl import __version__
from akl.api.deps import AppState, JobRegistry
from akl.api.main import create_app
from akl.api.middleware.ratelimit import TokenBucketLimiter
from akl.config import RetrievalSettings, Settings
from akl.rag.context_builder import ContextBuilder
from akl.rag.query import QueryProcessor
from akl.rag.retrieval.engine import RetrievalResult, RetrievalUnavailableError
from akl.rag.retrieval.models import Candidate
from akl.rag.service import AnswerResponse, SearchResponse
from akl.security.auth import Authenticator
from akl.security.principal import Principal

pytestmark = pytest.mark.unit


class FakeRAG:
    """Stands in for RAGService: deterministic results built from the shared fixture rows."""

    def __init__(self) -> None:
        self.flags: list[str] = []
        self.processor = QueryProcessor(RetrievalSettings())
        self.calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None
        self.qdrant = (
            None  # dense unavailable in the fake; readiness still ok because sparse is present
        )
        self.bm25 = type("FakeIndex", (), {"version": "fake-v1"})()

    def _retrieval(self, k: int = 3) -> RetrievalResult:
        cands = []
        for i, r in enumerate(rows()[:k]):
            c = Candidate(
                r["chunk_id"],
                r,
                rrf_score=0.1 - i * 0.01,
                dense_score=0.9 - i * 0.1,
                dense_rank=i + 1,
            )
            c.rerank_score = 0.9 - i * 0.1
            cands.append(c)
        return RetrievalResult(
            cands,
            0.9,
            True,
            None,
            [c.chunk_id for c in cands],
            [cands[0].chunk_id],
            [c.chunk_id for c in cands],
            [],
            {"dense": 3.0, "total": 5.0},
            "lexical",
        )

    def search(self, text: str, principal: Principal, **kw: Any) -> SearchResponse:
        if self.fail_with:
            raise self.fail_with
        self.calls.append({"op": "search", "text": text, "principal": principal.subject, **kw})
        q = self.processor.process(text, principal, filters=kw.get("filters"))
        r = self._retrieval(kw.get("k") or 3)
        results = [
            {
                "rank": i + 1,
                "chunk_id": c.chunk_id,
                "lineage_id": c.payload["lineage_id"],
                "document_id": c.payload["document_id"],
                "title": c.payload["title"],
                "source_type": "markdown",
                "heading_breadcrumb": c.payload["heading_breadcrumb"],
                "chunk_type": "prose",
                "locator": "x.md#L1-L2",
                "url": c.payload["source_uri"],
                "scores": c.scores(),
                **({"text": c.text} if kw.get("include_text", True) else {}),
            }
            for i, c in enumerate(r.candidates)
        ]
        return SearchResponse(
            kw.get("request_id") or "rid", "tid", q, r, results, {"total": 5.0}, "snap"
        )

    def _answer(self, text: str, principal: Principal, **kw: Any) -> AnswerResponse:
        q = self.processor.process(text, principal)
        r = self._retrieval()
        ctx = ContextBuilder(lambda t: len(t.split()), budget_tokens=1000, top_k=8).build(
            r.candidates
        )
        from akl.rag.citations import extractive_answer

        cited = extractive_answer(ctx, passages=2)
        return AnswerResponse(
            kw.get("request_id") or "rid",
            "tid",
            cited.mode,
            cited.answer,
            [c.as_dict() for c in cited.citations],
            0.9,
            None,
            ["deduplicated"],
            ctx,
            r,
            q,
            {"total": 7.0},
            str(kw.get("conversation_id") or uuid.uuid4()),
            None,
            None,
            {},
        )

    def answer(self, text: str, principal: Principal, **kw: Any) -> AnswerResponse:
        self.calls.append({"op": "answer", "text": text, **kw})
        return self._answer(text, principal, **kw)

    def stream_answer(self, text: str, principal: Principal, **kw: Any) -> Iterator[dict[str, Any]]:
        resp = self._answer(text, principal, **kw)
        common = {
            "request_id": resp.request_id,
            "trace_id": resp.trace_id,
            "conversation_id": resp.conversation_id,
        }
        yield {
            "event": "meta",
            **common,
            "intent": resp.query.intent.value,
            "confidence": 0.9,
            "sufficient": True,
        }
        for piece in (resp.answer or "").split(" "):
            yield {"event": "token", "text": piece + " "}
        yield {"event": "citations", **common, "citations": resp.citations}
        yield {
            "event": "done",
            **common,
            "mode": resp.mode,
            "confidence": 0.9,
            "reason": None,
            "flags": resp.flags,
            "timings_ms": resp.timings_ms,
            "response": resp,
        }


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key in ("AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"):
        monkeypatch.setenv(key, "x")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings.load(config_file=None, env_file=None)


def _state(settings: Settings, rag: Any, *, rpm: int = 1000) -> AppState:
    return AppState(
        settings=settings,
        engine=None,
        db=None,
        rag=rag,
        authenticator=Authenticator(settings, None),
        limiter=TokenBucketLimiter({"search": rpm, "chat": rpm}, default_rpm=rpm),
        jobs=JobRegistry(),
        version=__version__,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, FakeRAG, AppState]]:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="true")
    rag = FakeRAG()
    state = _state(settings, rag)
    with TestClient(create_app(settings, state=state)) as c:
        yield c, rag, state


def test_health_live_ready_and_metrics(client: tuple[TestClient, FakeRAG, AppState]) -> None:
    c, _rag, state = client
    live = c.get("/v1/health")
    assert live.status_code == 200
    assert live.json()["version"] == __version__
    assert "X-Request-ID" in live.headers
    ready = c.get("/v1/health/ready")
    assert ready.status_code == 200  # no db → no required probes fail; rag present
    state.rag = None
    assert c.get("/v1/health/ready").status_code == 503
    state.rag = _rag
    m = c.get("/metrics")
    assert m.status_code == 200
    assert "akl_http_requests_total" in m.text


def test_search_contract_request_id_and_filters(
    client: tuple[TestClient, FakeRAG, AppState],
) -> None:
    c, rag, _ = client
    rid = uuid.uuid4().hex
    r = c.post(
        "/v1/search",
        json={
            "query": "nightly backups",
            "k": 2,
            "filters": {"source_type": ["markdown"], "repo": ["org/docs"]},
            "include_text": False,
        },
        headers={"X-Request-ID": rid},
    )
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == rid
    body = r.json()
    assert body["request_id"] == rid
    assert len(body["results"]) == 2
    assert body["results"][0]["rank"] == 1
    assert "text" not in body["results"][0] or body["results"][0]["text"] is None
    assert body["query"]["intent"] == "factual_lookup"
    assert body["query"]["filters_applied"]["hard"]["source_types"] == ["markdown"]
    assert body["sufficient"] is True
    assert rag.calls[-1]["filters"].repos == ["org/docs"]
    bad = c.post("/v1/search", json={"query": "", "mode": "hybrid"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "AKL-E6002"
    unknown = c.post("/v1/search", json={"query": "x", "bogus": 1})
    assert unknown.status_code == 422


def test_error_envelope_for_backend_failure(client: tuple[TestClient, FakeRAG, AppState]) -> None:
    c, rag, _ = client
    rag.fail_with = RetrievalUnavailableError("all backends down", details={"errors": ["x"]})
    r = c.post("/v1/search", json={"query": "anything"})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "AKL-E6012"
    assert err["retryable"] is True
    assert err["request_id"] == r.headers["X-Request-ID"]


def test_chat_non_stream_and_stream(client: tuple[TestClient, FakeRAG, AppState]) -> None:
    c, _rag, _ = client
    r = c.post("/v1/chat", json={"query": "how are backups taken", "include_trace": True})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    assert body["citations"]
    assert body["citations"][0]["index"] == 1
    assert body["retrieval"]["intent"] == "factual_lookup"
    assert body["trace"]["confidence"] == 0.9
    assert uuid.UUID(body["conversation_id"])

    with c.stream("POST", "/v1/chat", json={"query": "how are backups taken", "stream": True}) as s:
        assert s.status_code == 200
        assert s.headers["content-type"].startswith("text/event-stream")
        raw = "".join(s.iter_text())
    events = [blk.split("\n", 1) for blk in raw.strip().split("\n\n")]
    kinds = [e[0].removeprefix("event: ") for e in events]
    assert kinds[0] == "meta"
    assert kinds[-2] == "citations"
    assert kinds[-1] == "done"
    assert kinds.count("token") >= 2
    done = json.loads(events[-1][1].removeprefix("data: "))
    assert done["mode"] == "extractive"
    assert "response" not in done  # internal object never leaks

    bad = c.post("/v1/chat", json={"query": "x", "conversation_id": "not-a-uuid"})
    assert bad.status_code == 404
    assert bad.json()["error"]["code"] == "AKL-E6030"


def test_auth_jwt_scopes_and_api_key_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="false", AKL_JWT_SECRET="s" * 40)
    rag = FakeRAG()
    state = _state(settings, rag)
    auth = state.authenticator
    with TestClient(create_app(settings, state=state)) as c:
        assert c.post("/v1/search", json={"query": "x"}).status_code == 401
        assert c.post("/v1/search", json={"query": "x"}).json()["error"]["code"] == "AKL-E1001"
        reader = auth.mint_token(
            "alice", groups=["eng"], security_levels=["public", "internal"], roles=["reader"]
        )
        ok = c.post(
            "/v1/search", json={"query": "x"}, headers={"Authorization": f"Bearer {reader}"}
        )
        assert ok.status_code == 200
        assert rag.calls[-1]["principal"] == "alice"
        denied = c.post(
            "/v1/documents",
            files={"files": ("a.md", b"# hi", "text/markdown")},
            headers={"Authorization": f"Bearer {reader}"},
        )
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "AKL-E1003"
        assert (
            c.post(
                "/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer garbage"}
            ).json()["error"]["code"]
            == "AKL-E1002"
        )
        # This fake AppState has no database, so any X-API-Key hits the "store unavailable" branch
        # (not "wrong key") — that's now correctly a 503, not a 401 (see AuthError vs
        # ApiKeyStoreUnavailableError). A real "wrong key -> 401" check lives in the live
        # component test, where a database is actually available to check the key against.
        no_db_resp = c.post("/v1/search", json={"query": "x"}, headers={"X-API-Key": "nope"})
        assert no_db_resp.status_code == 503
        assert no_db_resp.json()["error"]["code"] == "AKL-E1007"
        principal = auth.verify_token(reader)
        assert principal.has_scope("chat:write") and not principal.has_scope("documents:write")  # noqa: PT018
        health = c.get("/v1/health")
        assert health.status_code == 200  # health never requires auth


def test_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="true")
    state = _state(settings, FakeRAG(), rpm=2)
    with TestClient(create_app(settings, state=state)) as c:
        assert c.post("/v1/search", json={"query": "a"}).status_code == 200
        assert c.post("/v1/search", json={"query": "b"}).status_code == 200
        limited = c.post("/v1/search", json={"query": "c"})
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "AKL-E1006"
        assert int(limited.headers["Retry-After"]) >= 1
        assert (
            c.post("/v1/chat", json={"query": "d"}).status_code == 200
        )  # separate bucket per route class


def test_openapi_and_docs_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="true")
    state = _state(settings, FakeRAG())
    with TestClient(create_app(settings, state=state)) as c:
        spec = c.get("/openapi.json").json()
        paths = set(spec["paths"])
        assert {
            "/v1/search",
            "/v1/chat",
            "/v1/documents",
            "/v1/documents/{document_id}",
            "/v1/health/ready",
            "/v1/sources/github/sync",
            "/v1/jobs/{run_id}",
        } <= paths
        assert "/metrics" not in paths
    off = _settings(monkeypatch, AKL_AUTH_DISABLED="true", AKL_OPENAPI_ENABLED="false")
    with TestClient(create_app(off, state=_state(off, FakeRAG()))) as c:
        assert c.get("/openapi.json").status_code == 404


# --------------------------------------------------------------------------- admin: RBAC, keys, audit, GDPR (Batch H)
def test_admin_permissions_requires_scope_and_reports_not_found_without_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="false", AKL_JWT_SECRET="s" * 40)
    auth = Authenticator(settings, None)
    state = _state(settings, FakeRAG())
    with TestClient(create_app(settings, state=state)) as c:
        reader = auth.mint_token("bob", groups=[], security_levels=["public"], roles=["reader"])
        denied = c.patch(
            "/v1/admin/documents/00000000-0000-0000-0000-000000000000/permissions",
            json={"security_level": "internal"},
            headers={"Authorization": f"Bearer {reader}"},
        )
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "AKL-E1003"

        curator = auth.mint_token("carol", groups=[], security_levels=["public"], roles=["curator"])
        not_found = c.patch(
            "/v1/admin/documents/00000000-0000-0000-0000-000000000000/permissions",
            json={"security_level": "internal"},
            headers={"Authorization": f"Bearer {curator}"},
        )
        assert (
            not_found.status_code == 404
        )  # scope check passes; no database configured in this fake state
        assert not_found.json()["error"]["code"] == "AKL-E6030"

        bad_json = c.patch(
            "/v1/admin/documents/not-a-uuid/permissions",
            json={"bogus": True},
            headers={"Authorization": f"Bearer {curator}"},
        )
        assert bad_json.status_code == 422  # extra="forbid" rejects unknown fields


def test_admin_api_keys_reports_service_unavailable_not_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: 'no database configured' must be 503, never 401 (a client can't fix it by re-authenticating)."""
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="true")
    state = _state(settings, FakeRAG())
    with TestClient(create_app(settings, state=state)) as c:
        listed = c.get("/v1/admin/api-keys")
        assert listed.status_code == 503
        assert listed.json()["error"]["code"] == "AKL-E1007"
        created = c.post("/v1/admin/api-keys", json={"name": "x", "roles": ["reader"]})
        assert created.status_code == 503
        revoked = c.delete("/v1/admin/api-keys/00000000-0000-0000-0000-000000000000")
        assert revoked.status_code == 503


def test_admin_audit_query_empty_without_db_but_requires_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="false", AKL_JWT_SECRET="s" * 40)
    auth = Authenticator(settings, None)
    state = _state(settings, FakeRAG())
    with TestClient(create_app(settings, state=state)) as c:
        reader = auth.mint_token("bob", groups=[], security_levels=["public"], roles=["reader"])
        denied = c.get("/v1/admin/audit", headers={"Authorization": f"Bearer {reader}"})
        assert denied.status_code == 403
        curator = auth.mint_token("carol", groups=[], security_levels=["public"], roles=["curator"])
        ok = c.get("/v1/admin/audit", headers={"Authorization": f"Bearer {curator}"})
        assert ok.status_code == 200
        assert ok.json() == []  # no database configured -> empty, not an error


def test_admin_gdpr_self_service_vs_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, AKL_AUTH_DISABLED="false", AKL_JWT_SECRET="s" * 40)
    auth = Authenticator(settings, None)
    state = _state(settings, FakeRAG())
    with TestClient(create_app(settings, state=state)) as c:
        alice = auth.mint_token("alice", groups=[], security_levels=["public"], roles=["reader"])
        own = c.delete(
            "/v1/admin/gdpr/principals/alice", headers={"Authorization": f"Bearer {alice}"}
        )
        assert own.status_code == 503  # authorized (it's her own data); fails only on "no database"
        other = c.delete(
            "/v1/admin/gdpr/principals/bob", headers={"Authorization": f"Bearer {alice}"}
        )
        assert other.status_code == 403
        assert other.json()["error"]["code"] == "AKL-E1003"
        admin_tok = auth.mint_token("root", groups=[], security_levels=["public"], roles=["admin"])
        admin_on_behalf = c.get(
            "/v1/admin/gdpr/principals/bob/export", headers={"Authorization": f"Bearer {admin_tok}"}
        )
        assert admin_on_behalf.status_code == 503  # admin scope authorizes acting on bob's behalf
