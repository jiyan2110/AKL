"""Fire-and-forget audit logging for API routes (PRD §9.5).
...
"""

from __future__ import annotations

import ipaddress
from typing import Any

from fastapi import Request

from akl.api.deps import AppState
from akl.db.repositories.audit import AuditLogRepository
from akl.observability.logging import get_logger
from akl.security.principal import Principal

log = get_logger("akl.api.audit")


def _valid_ip(host: str | None) -> str | None:
    """None unless ``host`` parses as a real IPv4/IPv6 address (e.g. TestClient's "testclient")."""
    if not host:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return None
    return host


def audit(
    state: AppState,
    request: Request,
    *,
    principal: Principal,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str = "success",
    details: dict[str, Any] | None = None,
) -> None:
    if state.db is None or not state.settings.governance.audit_log_enabled:
        return
    request_id = str(getattr(request.state, "request_id", None) or "")
    try:
        with state.db.session() as s:
            AuditLogRepository(s).log(
                action=action,
                principal_id=principal.subject,
                resource_type=resource_type,
                resource_id=resource_id,
                request_id=request_id or None,
                ip=_valid_ip(request.client.host if request.client else None),
                user_agent=request.headers.get("user-agent"),
                outcome=outcome,
                details=details,
            )
    except Exception as exc:  # never let audit logging break the request
        log.warning("audit_write_failed", action=action, error=str(exc))
