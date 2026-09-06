"""``/v1/admin/api-keys`` — create, list, revoke API keys (PRD §9.2, §10.8)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from akl.api.audit import audit
from akl.api.deps import AppState, get_state, scoped
from akl.api.schemas.admin import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyOut
from akl.errors import AKLError
from akl.security.principal import Principal

router = APIRouter(prefix="/v1/admin/api-keys", tags=["admin"])


class ApiKeyNotFoundError(AKLError):
    code = "AKL-E9020"
    http_status = 404
    retryable = False


def _out(row: object) -> ApiKeyOut:
    return ApiKeyOut(
        key_id=str(row.key_id),  # type: ignore[attr-defined]
        prefix=row.prefix,  # type: ignore[attr-defined]
        name=row.name,  # type: ignore[attr-defined]
        scopes=list(row.scopes),  # type: ignore[attr-defined]
        groups=list(row.groups),  # type: ignore[attr-defined]
        security_levels=list(row.security_levels),  # type: ignore[attr-defined]
        created_at=row.created_at,  # type: ignore[attr-defined]
        expires_at=row.expires_at,  # type: ignore[attr-defined]
        last_used_at=row.last_used_at,  # type: ignore[attr-defined]
        revoked_at=row.revoked_at,  # type: ignore[attr-defined]
    )


@router.get("", response_model=list[ApiKeyOut])
def list_keys(
    state: AppState = Depends(get_state), _: Principal = Depends(scoped("keys:manage"))
) -> list[ApiKeyOut]:
    return [_out(k) for k in state.authenticator.list_api_keys()]


@router.post("", response_model=ApiKeyCreateResponse, status_code=201)
def create_key(
    body: ApiKeyCreateRequest,
    request: Request,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("keys:manage")),
) -> ApiKeyCreateResponse:
    minted = state.authenticator.create_api_key(
        name=body.name,
        groups=body.groups,
        security_levels=body.security_levels,
        roles=body.roles,
        expires_at=body.expires_at,
    )
    audit(
        state,
        request,
        principal=principal,
        action="api_key.create",
        resource_type="api_key",
        resource_id=str(minted.key_id),
        details={"name": body.name, "roles": body.roles},
    )
    return ApiKeyCreateResponse(key_id=str(minted.key_id), prefix=minted.prefix, token=minted.token)


@router.delete("/{key_id}", status_code=204)
def revoke_key(
    key_id: str,
    request: Request,
    state: AppState = Depends(get_state),
    principal: Principal = Depends(scoped("keys:manage")),
) -> None:
    try:
        kid = uuid.UUID(key_id)
    except ValueError as exc:
        raise ApiKeyNotFoundError("invalid key id", details={"key_id": key_id}) from exc
    revoked = state.authenticator.revoke_api_key(kid)
    if not revoked:
        raise ApiKeyNotFoundError("key not found or already revoked", details={"key_id": key_id})
    audit(
        state,
        request,
        principal=principal,
        action="api_key.revoke",
        resource_type="api_key",
        resource_id=key_id,
    )
