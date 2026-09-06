# Security Overview

## Authentication
- **JWT (HS256)** — `Authenticator.mint_token()`/`verify_token()` (`akl/security/auth.py`).
  Requires `AKL_JWT_SECRET`; missing it is a server misconfiguration (`AKL-E1008`, 503), never a
  401 (see [ADR-014](../adr/ADR-014-service-errors-vs-auth-errors.md)).
- **API keys** — `akl_<prefix>_<secret>`, hashed (HMAC) at rest, never stored or logged in
  plaintext. Individually revocable via `DELETE /v1/admin/api-keys/{key_id}`.
- **Dev bypass** — `AKL_AUTH_DISABLED=true` is refused outside `AKL_ENV=dev` (enforced in
  `Settings`'s cross-section validation, not just a docs warning).

## Authorization (RBAC)
Roles → scopes are defined in `ROLE_SCOPES` (`akl/security/auth.py`):

| Role | Scopes |
|---|---|
| `reader` | `search:read`, `chat:write` |
| `contributor` | + `documents:write` |
| `curator` | + `quarantine:manage`, `documents:delete`, `documents:permissions`, `keys:manage`, `audit:read` |
| `admin` | `*` (everything) |
| `service` | `admin:reload`, `pipelines:trigger` (for automation, e.g. the Airflow→API reload notification) |

Document-level access is a separate, per-document check (`Principal.can_read`): a chunk is visible
only if the principal's `security_levels` includes the document's level **and** either the
document has no `allowed_groups` or the principal's groups intersect them. This applies uniformly
to search results, chat citations, and document/lineage endpoints.

## Data protection
- **PII scanning** (`akl/governance/pii.py`) runs on every parsed document; only
  `sha256(matched_value)` is ever persisted (`pii_mentions.value_hash`) — the raw match never
  reaches the database or logs.
- **GDPR erasure/export** (`akl/governance/gdpr.py`) is scoped to conversation history — the only
  per-principal personal data this system holds. Self-service by default
  (`DELETE /v1/admin/gdpr/principals/{me}`); acting on someone else's data requires the
  `gdpr:manage` scope (admin only).
- **Audit log** (`akl/db/repositories/audit.py`) is append-only, retained 400 days by default
  (`governance.audit_log_retention_days`), and records permission changes, API key lifecycle
  events, and both soft and hard document deletes.
- **Hard delete** additionally purges the raw Bronze object from object storage and the
  document's `pii_mentions` rows — the two things that are not compaction-eligible (unlike
  Silver/Gold Parquet, which a tombstone + the next compaction pass removes).

## Logging
Structured logs (`akl/observability/logging.py`) redact anything matching a secret-like key
(`password`, `token`, `api_key`, `authorization`, ...), bearer tokens, and credentials embedded in
URLs, before they reach any sink — verified in `tests/unit/test_observability.py`.

## Transport
Production deploys (`docker-compose.prod.yml`) sit behind Traefik with TLS (Let's Encrypt or
self-signed for internal/staging use — `docker/traefik/gen-selfsigned.sh`), including HSTS headers
on the API router. Internal services (Postgres, MinIO, Qdrant, Prometheus, etc.) publish no host
ports in the production overlay.

## Known gaps
- No SSO/OIDC integration — only local JWT and API-key auth.
- Alerting has no receiver configured by default (see the [Alerting runbook](../runbooks/alerting.md)) —
  a security-relevant alert firing does not by itself notify anyone.
- No automated dependency/CVE scanning is wired into CI yet.
