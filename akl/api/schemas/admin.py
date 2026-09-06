"""Admin API schemas: permissions, API keys, audit, GDPR, PII (PRD §9.5-9.8, §10.8)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from akl.api.schemas.common import StrictModel


class PermissionsUpdate(StrictModel):
    security_level: Literal["public", "internal", "restricted"] | None = None
    allowed_groups: list[str] | None = None


class PermissionsResponse(StrictModel):
    document_id: str
    security_level: str
    allowed_groups: list[str]


class ApiKeyOut(StrictModel):
    key_id: str
    prefix: str
    name: str | None
    scopes: list[str]
    groups: list[str]
    security_levels: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    groups: list[str] = Field(default_factory=list)
    security_levels: list[str] = Field(default_factory=lambda: ["public", "internal"])
    roles: list[str] = Field(default_factory=lambda: ["reader"])
    expires_at: datetime | None = None


class ApiKeyCreateResponse(StrictModel):
    key_id: str
    prefix: str
    token: str  # shown once


class AuditEntry(StrictModel):
    id: int
    ts: datetime
    principal_id: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    request_id: str | None
    outcome: str | None
    details: dict[str, Any] = Field(default_factory=dict)


class PiiMentionOut(StrictModel):
    pii_type: str
    count: int


class PiiSummaryResponse(StrictModel):
    document_id: str
    counts: dict[str, int]


class GdprErasureResponse(StrictModel):
    principal_id: str
    conversations_deleted: int
    messages_deleted: int
    citations_deleted: int


class GdprExportResponse(StrictModel):
    principal_id: str
    conversations: list[dict[str, Any]]
    truncated: bool
