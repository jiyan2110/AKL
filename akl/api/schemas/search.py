"""Search and chat request/response models (PRD §10.5, §10.6)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from akl.api.schemas.common import StrictModel


class SearchFilters(StrictModel):
    source_type: list[Literal["pdf", "markdown", "html", "github"]] = Field(default_factory=list)
    repo: list[str] = Field(default_factory=list)
    chunk_type: list[str] = Field(default_factory=list)
    code_language: list[str] = Field(default_factory=list)
    document_id: list[str] = Field(default_factory=list)
    updated_after: datetime | None = None


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    k: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters | None = None
    rerank: bool = True
    include_text: bool = True
    precision: Literal["default", "high"] = "default"


class Scores(StrictModel):
    dense: float | None = None
    sparse: float | None = None
    rrf: float | None = None
    rerank: float | None = None


class SearchResult(StrictModel):
    rank: int
    chunk_id: str
    lineage_id: str | None = None
    document_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    locator: str
    url: str | None = None
    heading_breadcrumb: str | None = None
    chunk_type: str | None = None
    text: str | None = None
    scores: Scores


class QueryInfo(StrictModel):
    normalized: str
    corrected: str
    intent: str
    entities: dict[str, list[str]]
    filters_applied: dict[str, Any]


class SearchResponse(StrictModel):
    request_id: str
    trace_id: str
    results: list[SearchResult]
    query: QueryInfo
    confidence: float
    sufficient: bool
    flags: list[str]
    timings_ms: dict[str, float]
    gold_snapshot_id: str | None = None


class ChatRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    stream: bool = False
    filters: SearchFilters | None = None
    k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["auto", "generative", "extractive"] = "auto"
    include_trace: bool = False


class Citation(StrictModel):
    index: int
    chunk_id: str
    lineage_id: str | None = None
    document_id: str
    title: str | None = None
    source_type: str | None = None
    locator: str
    url: str | None = None
    snippet: str
    score: float | None = None


class ChatResponse(StrictModel):
    request_id: str
    trace_id: str
    conversation_id: str | None
    mode: str
    answer: str | None
    citations: list[Citation]
    confidence: float
    reason: str | None
    flags: list[str]
    retrieval: dict[str, Any]
    llm: dict[str, Any] | None = None
    timings_ms: dict[str, float]
    trace: dict[str, Any] | None = None


class ConversationTurn(StrictModel):
    turn: int
    role: str
    content: str
    mode: str | None = None
    confidence: float | None = None
    created_at: datetime


class ConversationResponse(StrictModel):
    conversation_id: str
    summary: str | None
    turn_count: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    turns: list[ConversationTurn]
