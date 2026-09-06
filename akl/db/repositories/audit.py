"""AuditLogRepository — writes/reads ``audit_log`` (PRD §9.5, Appendix A.11).

Append-only: every write is a new row (the table is RANGE-partitioned by ``ts``). Failures to
write an audit row must never fail the request that triggered it — callers should catch and log,
not propagate (see ``akl.api.audit.audit`` helper).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from akl.db.models import AuditLog
from akl.db.repositories import Repository


class AuditLogRepository(Repository):
    def log(
        self,
        *,
        action: str,
        principal_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        request_id: str | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.session.add(
            AuditLog(
                principal_id=principal_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                request_id=request_id,
                ip=ip,
                user_agent=user_agent,
                outcome=outcome,
                details=details or {},
            )
        )

    def query(
        self,
        *,
        principal_id: str | None = None,
        action: str | None = None,
        resource_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.ts.desc()).limit(limit)
        if principal_id:
            stmt = stmt.where(AuditLog.principal_id == principal_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if resource_id:
            stmt = stmt.where(AuditLog.resource_id == resource_id)
        if since:
            stmt = stmt.where(AuditLog.ts >= since)
        if until:
            stmt = stmt.where(AuditLog.ts <= until)
        return list(self.session.scalars(stmt))
