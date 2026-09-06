"""``/v1/admin/gdpr`` — erase or export a principal's own conversation data (PRD §9.7).

Any authenticated principal may erase or export their *own* data (the actual GDPR right); an
admin (scope ``gdpr:manage``) may act on behalf of any principal for a routed compliance request.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from akl.api.audit import audit
from akl.api.deps import AppState, get_principal, get_state
from akl.api.schemas.admin import GdprErasureResponse, GdprExportResponse
from akl.errors import AKLError
from akl.governance.gdpr import erase_principal, export_principal
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin/gdpr", tags=["admin"])


class GdprUnavailableError(AKLError):
    code = "AKL-E9030"
    http_status = 503
    retryable = True


class GdprForbiddenError(AKLError):
    code = "AKL-E1003"
    http_status = 403
    retryable = False


def _authorize(principal: Principal, target_principal_id: str) -> None:
    if principal.subject == target_principal_id or principal.has_scope("gdpr:manage"):
        return
    raise GdprForbiddenError(
        "may only act on your own data (or hold scope 'gdpr:manage')",
        details={"target": target_principal_id},
    )


@router.delete("/principals/{principal_id}", response_model=GdprErasureResponse)
def erase(
    principal_id: str,
    request: Request,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(get_principal),
) -> GdprErasureResponse:
    _authorize(principal, principal_id)
    if state.db is None:
        raise GdprUnavailableError("no database configured")
    with state.db.session() as s:
        result = erase_principal(s, principal_id)
    audit(
        state,
        request,
        principal=principal,
        action="gdpr.erase",
        resource_type="principal",
        resource_id=principal_id,
        details=result.as_dict(),
    )
    return GdprErasureResponse(**result.as_dict())


@router.get("/principals/{principal_id}/export", response_model=GdprExportResponse)
def export(
    principal_id: str,
    request: Request,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(get_principal),
) -> GdprExportResponse:
    _authorize(principal, principal_id)
    if state.db is None:
        raise GdprUnavailableError("no database configured")
    with state.db.session() as s:
        result = export_principal(
            s,
            principal_id,
            max_conversations=state.settings.governance.gdpr_export_max_conversations,
        )
    audit(
        state,
        request,
        principal=principal,
        action="gdpr.export",
        resource_type="principal",
        resource_id=principal_id,
        details={"conversations": len(result["conversations"])},
    )
    return GdprExportResponse(**result)
