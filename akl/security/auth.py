"""Authentication (PRD §9.2): JWT bearer tokens (HS256, MVP), API keys, dev bypass."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from sqlalchemy import select

from akl.config import Settings
from akl.db.models import ApiKey
from akl.db.session import Database
from akl.errors import AKLError
from akl.security.principal import Principal

ROLE_SCOPES: dict[str, frozenset[str]] = {
    "reader": frozenset({"search:read", "chat:write"}),
    "contributor": frozenset({"search:read", "chat:write", "documents:write"}),
    "curator": frozenset(
        {
            "search:read",
            "chat:write",
            "documents:write",
            "quarantine:manage",
            "documents:delete",
            "documents:permissions",
            "keys:manage",
            "audit:read",
        }
    ),
    "admin": frozenset({"*"}),
    "service": frozenset({"admin:reload", "pipelines:trigger"}),
}


class AuthError(AKLError):
    code = "AKL-E1001"
    http_status = 401
    retryable = False


class InvalidTokenError(AuthError):
    code = "AKL-E1002"


class ForbiddenError(AKLError):
    code = "AKL-E1003"
    http_status = 403
    retryable = False


class ApiKeyStoreUnavailableError(AKLError):
    """No database configured for API-key storage (AKL-E1007) — a service-availability issue,
    not a credentials problem, so it must not be reported as 401 (which tells the client to
    retry with different credentials)."""

    code = "AKL-E1007"
    http_status = 503
    retryable = True


class AuthConfigError(AKLError):
    """The server has no signing secret configured (AKL-E1008) — an operator misconfiguration,
    not something a client can fix by presenting different credentials."""

    code = "AKL-E1008"
    http_status = 503
    retryable = True


class InvalidApiKeyError(AuthError):
    code = "AKL-E1005"


@dataclass(frozen=True)
class MintedKey:
    key_id: uuid.UUID
    prefix: str
    secret: str  # full key shown once: akl_<prefix>_<secret>

    @property
    def token(self) -> str:
        return f"akl_{self.prefix}_{self.secret}"


def scopes_for_roles(roles: list[str] | tuple[str, ...]) -> frozenset[str]:
    out: set[str] = set()
    for role in roles:
        out |= ROLE_SCOPES.get(role, frozenset())
    return frozenset(out)


class Authenticator:
    def __init__(self, settings: Settings, db: Database | None) -> None:
        self.settings = settings
        self.db = db
        self._secret = (
            settings.api.jwt_secret.get_secret_value() if settings.api.jwt_secret else None
        )
        self._pepper = (
            settings.api.api_key_pepper.get_secret_value() if settings.api.api_key_pepper else ""
        )

    @property
    def disabled(self) -> bool:
        return self.settings.api.auth_disabled and self.settings.core.env.value == "dev"

    # -- JWT ----------------------------------------------------------------------------
    def mint_token(
        self,
        subject: str,
        *,
        groups: list[str],
        security_levels: list[str],
        roles: list[str],
        ttl_s: int | None = None,
    ) -> str:
        if not self._secret:
            raise AuthConfigError("AKL_JWT_SECRET not configured")
        now = datetime.now(UTC)
        payload: dict[str, Any] = {
            "iss": self.settings.api.jwt_issuer,
            "aud": self.settings.api.jwt_audience,
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=ttl_s or self.settings.api.jwt_ttl_s)).timestamp()),
            "groups": groups,
            "security_levels": security_levels,
            "roles": roles,
            "scope": " ".join(sorted(scopes_for_roles(roles))),
        }
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def verify_token(self, token: str) -> Principal:
        if not self._secret:
            raise AuthConfigError("AKL_JWT_SECRET not configured")
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=self.settings.api.jwt_audience,
                issuer=self.settings.api.jwt_issuer,
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(f"invalid token: {exc}") from exc
        scopes = frozenset(str(claims.get("scope", "")).split()) or scopes_for_roles(
            list(claims.get("roles", ["reader"]))
        )
        return Principal(
            subject=str(claims["sub"]),
            groups=frozenset(str(g) for g in claims.get("groups", [])),
            security_levels=frozenset(str(s) for s in claims.get("security_levels", ["public"])),
            scopes=scopes,
        )

    # -- API keys -------------------------------------------------------------------------
    def hash_key(self, secret: str) -> str:
        return hmac.new(
            self._pepper.encode("utf-8"), secret.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def create_api_key(
        self,
        *,
        name: str,
        groups: list[str],
        security_levels: list[str],
        roles: list[str],
        owner_user_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> MintedKey:
        if self.db is None:
            raise ApiKeyStoreUnavailableError("database required for API keys")
        prefix = secrets.token_hex(4)
        secret = secrets.token_urlsafe(32)
        key = ApiKey(
            key_id=uuid.uuid4(),
            prefix=prefix,
            key_hash=self.hash_key(secret),
            name=name,
            owner_user_id=owner_user_id,
            scopes=sorted(scopes_for_roles(roles)),
            groups=groups,
            security_levels=security_levels,
            expires_at=expires_at,
        )
        with self.db.session() as s:
            s.add(key)
        return MintedKey(key.key_id, prefix, secret)

    def verify_api_key(self, token: str) -> Principal:
        if self.db is None:
            raise ApiKeyStoreUnavailableError("database required for API keys")
        parts = token.split("_", 2)
        if len(parts) != 3 or f"{parts[0]}_" != self.settings.api.api_key_prefix:
            raise InvalidApiKeyError("malformed API key")
        _, prefix, secret = parts
        with self.db.session() as s:
            row = s.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
            if row is None or row.revoked_at is not None:
                raise InvalidApiKeyError("unknown or revoked API key")
            if row.expires_at is not None and row.expires_at < datetime.now(UTC):
                raise InvalidApiKeyError("expired API key")
            if not hmac.compare_digest(row.key_hash, self.hash_key(secret)):
                raise InvalidApiKeyError("invalid API key")
            row.last_used_at = datetime.now(UTC)
            return Principal(
                subject=f"apikey:{row.name or prefix}",
                groups=frozenset(row.groups),
                security_levels=frozenset(row.security_levels),
                scopes=frozenset(row.scopes),
            )

    def list_api_keys(self) -> list[ApiKey]:
        if self.db is None:
            raise ApiKeyStoreUnavailableError("database required for API keys")
        with self.db.session() as s:
            return list(s.scalars(select(ApiKey).order_by(ApiKey.created_at.desc())))

    def revoke_api_key(self, key_id: uuid.UUID) -> bool:
        if self.db is None:
            raise ApiKeyStoreUnavailableError("database required for API keys")
        with self.db.session() as s:
            row = s.get(ApiKey, key_id)
            if row is None or row.revoked_at is not None:
                return False
            row.revoked_at = datetime.now(UTC)
            return True

    # -- resolution ---------------------------------------------------------------------------
    def authenticate(self, *, authorization: str | None, api_key: str | None) -> Principal:
        if api_key:
            return self.verify_api_key(api_key)
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise InvalidTokenError("expected 'Authorization: Bearer <jwt>'")
            return self.verify_token(token)
        if self.disabled:
            return Principal.dev()
        raise AuthError("missing credentials")


def require_scope(principal: Principal, scope: str) -> None:
    if not principal.has_scope(scope):
        raise ForbiddenError(f"scope {scope!r} required", details={"subject": principal.subject})
