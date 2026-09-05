"""Document, upload and source models (PRD §10.3, §10.4)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from akl.api.schemas.common import StrictModel


class UploadItem(StrictModel):
    filename: str
    document_id: str
    content_sha256: str
    status: str
    deduplicated: bool
    size_bytes: int


class UploadResponse(StrictModel):
    items: list[UploadItem]
    run_id: str
    processing: Literal["async", "sync"]
    status_url: str


class DocumentSummary(StrictModel):
    document_id: str
    canonical_source_uri: str
    source_type: str
    connector_id: str
    title: str | None
    status: str
    security_level: str
    allowed_groups: list[str]
    is_duplicate_of: str | None
    latest_content_sha256: str | None
    created_at: datetime
    updated_at: datetime


class DocumentVersionOut(StrictModel):
    document_version_id: str
    content_sha256: str
    parser_name: str | None
    parser_version: str
    quality_score: float | None
    language: str | None
    word_count: int | None
    parsed_at: datetime | None
    fetched_at: datetime | None


class DocumentDetail(DocumentSummary):
    current_version_id: str | None
    versions: list[DocumentVersionOut]
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkOut(StrictModel):
    chunk_id: str
    chunk_index: int
    chunk_type: str
    heading_breadcrumb: str | None
    text: str
    token_count: int | None
    quality_score: float | None
    page_start: int | None = None
    line_start: int | None = None


class DeleteResponse(StrictModel):
    document_id: str
    mode: Literal["soft", "hard"]
    status: str
    chunks_tombstoned: int


class GitHubSyncRequest(StrictModel):
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    branch: str | None = None
    full: bool = False


class SourceStatus(StrictModel):
    id: str
    type: str
    enabled: bool
    healthy: bool
    detail: str
    documents_count: int
    last_run_id: str | None
    last_success_at: datetime | None


class TriggerResponse(StrictModel):
    run_id: str
    status_url: str
    message: str


class JobStatus(StrictModel):
    run_id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    stages: list[str]
    report: dict[str, Any] | None = None
    errors: list[dict[str, str]] = Field(default_factory=list)
