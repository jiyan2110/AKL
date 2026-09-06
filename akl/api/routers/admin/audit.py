"""``GET /v1/admin/audit`` — query the audit log (PRD §9.5, §10.8)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from akl.api.deps import AppState, get_state, scoped
from akl.api.schemas.admin import AuditEntry
from akl.db.repositories.audit import AuditLogRepository
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin/audit", tags=["admin"])


@router.get("", response_model=list[AuditEntry])
def query_audit(
    principal_id: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    resource_id: Annotated[str | None, Query()] = None,
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    state: AppState = Depends(get_state),
    _: Principal = Depends(scoped("audit:read")),
) -> list[AuditEntry]:
    if state.db is None:
        return []
    with state.db.session() as s:
        rows = AuditLogRepository(s).query(
            principal_id=principal_id,
            action=action,
            resource_id=resource_id,
            since=since,
            until=until,
            limit=limit,
        )
        return [
            AuditEntry(
                id=r.id,
                ts=r.ts,
                principal_id=r.principal_id,
                action=r.action,
                resource_type=r.resource_type,
                resource_id=r.resource_id,
                request_id=r.request_id,
                outcome=r.outcome,
                details=dict(r.details or {}),
            )
            for r in rows
        ]
