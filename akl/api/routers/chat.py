"""Chat API with SSE streaming and conversations (PRD §10.6, §6.9, §6.8)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from akl.api import metrics
from akl.api.deps import AppState, get_request_id, get_state, rate_limited, scoped
from akl.api.routers.search import to_filters
from akl.api.schemas.search import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationResponse,
    ConversationTurn,
)
from akl.db.repositories.conversations import ConversationRepository
from akl.errors import AKLError
from akl.rag.memory import ConversationNotFoundError
from akl.security.principal import Principal

router = APIRouter(prefix="/v1", tags=["chat"])


def _conversation_uuid(value: str | None) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ConversationNotFoundError(uuid.UUID(int=0)) from exc


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _record_metrics(
    mode: str, stream: bool, flags: list[str], citations: int, timings: dict[str, float]
) -> None:
    metrics.CHAT_REQUESTS.labels(mode=mode, stream=str(stream).lower()).inc()
    for flag in flags:
        metrics.ANSWER_FLAGS.labels(flag=flag).inc()
    metrics.ANSWER_CITATIONS.observe(citations)
    for phase in ("llm_first_token", "llm_total"):
        if phase in timings:
            metrics.LLM_LATENCY.labels(phase=phase.replace("llm_", "")).observe(
                timings[phase] / 1000.0
            )


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limited("chat"))],
    responses={200: {"content": {"text/event-stream": {}}}},
)
def chat(
    body: ChatRequest,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("chat:write")),
    request_id: str = Depends(get_request_id),
) -> Any:
    conv_id = _conversation_uuid(body.conversation_id)
    filters = to_filters(body.filters)
    if body.stream:
        return StreamingResponse(
            _stream(state, body, principal, request_id, conv_id, filters),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )
    with state.lock:
        res = state.rag.answer(
            body.query,
            principal,
            filters=filters,
            request_id=request_id,
            conversation_id=conv_id,
            mode=body.mode,
        )
    _record_metrics(res.mode, False, res.flags, len(res.citations), res.timings_ms)
    if not res.retrieval.sufficient:
        metrics.INSUFFICIENT.labels(reason=res.reason or "unknown").inc()
    payload = res.as_dict(include_trace=body.include_trace)
    payload["citations"] = [Citation(**c) for c in res.citations]
    return ChatResponse(**payload)


def _stream(
    state: AppState,
    body: ChatRequest,
    principal: Principal,
    request_id: str,
    conv_id: uuid.UUID | None,
    filters: Any,
) -> Iterator[str]:
    metrics.SSE_ACTIVE.inc()
    try:
        with state.lock:
            for event in state.rag.stream_answer(
                body.query,
                principal,
                filters=filters,
                request_id=request_id,
                conversation_id=conv_id,
                mode=body.mode,
            ):
                kind = event["event"]
                if kind == "done":
                    resp = event.pop("response")
                    _record_metrics(
                        resp.mode, True, resp.flags, len(resp.citations), resp.timings_ms
                    )
                    if body.include_trace:
                        event["trace"] = resp.retrieval.trace()
                    event["llm"] = (
                        {"model": resp.llm_model, **resp.usage} if resp.llm_model else None
                    )
                yield _sse(kind, event)
    except AKLError as exc:
        yield _sse("error", {"request_id": request_id, **exc.to_dict()})
    except Exception as exc:  # the stream has started; the client must see a terminal event
        yield _sse(
            "error",
            {
                "request_id": request_id,
                "code": "AKL-E9999",
                "message": f"{type(exc).__name__}: {exc}",
            },
        )
    finally:
        metrics.SSE_ACTIVE.dec()


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("chat:write")),
) -> ConversationResponse:
    cid = _conversation_uuid(conversation_id)
    assert cid is not None
    if state.db is None:
        raise ConversationNotFoundError(cid)
    with state.db.session() as s:
        repo = ConversationRepository(s)
        conv = repo.get_owned(cid, principal.subject)
        if conv is None:
            raise ConversationNotFoundError(cid)
        turns = repo.messages(cid)
        return ConversationResponse(
            conversation_id=str(conv.conversation_id),
            summary=conv.summary,
            turn_count=conv.turn_count,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            expires_at=conv.expires_at,
            turns=[
                ConversationTurn(
                    turn=m.turn,
                    role=m.role,
                    content=m.content,
                    mode=m.mode,
                    confidence=m.confidence,
                    created_at=m.created_at,
                )
                for m in turns
            ],
        )


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("chat:write")),
) -> None:
    cid = _conversation_uuid(conversation_id)
    assert cid is not None
    if state.db is None:
        raise ConversationNotFoundError(cid)
    with state.db.session() as s:
        repo = ConversationRepository(s)
        if repo.get_owned(cid, principal.subject) is None:
            raise ConversationNotFoundError(cid)
        repo.delete(cid)
