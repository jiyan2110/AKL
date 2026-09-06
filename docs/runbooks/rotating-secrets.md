# Runbook: Rotating Secrets

## `AKL_JWT_SECRET`
Rotating invalidates every outstanding JWT immediately (API keys are unaffected — they're a
separate credential). There is no dual-secret grace period built in; plan for a short window
where active sessions need to re-authenticate.

1. Generate a new value (32+ random characters).
2. Update `AKL_JWT_SECRET` in `.env` (dev) or `.env.prod` (prod) and restart `akl-api`.
3. Re-mint any service tokens your automation depends on: `akl-cli auth mint-token`.

## API keys
API keys are individually revocable and don't need a "rotate everything" story:

```bash
curl -X POST /v1/admin/api-keys -H "Authorization: Bearer <admin-jwt>" \
  -d '{"name": "ci-pipeline", "roles": ["contributor"]}'   # mint the replacement first
curl -X DELETE /v1/admin/api-keys/<old-key-id> -H "Authorization: Bearer <admin-jwt>"  # then revoke the old one
```

Revoking sets `revoked_at`; the key row is kept (for audit purposes), never hard-deleted.

## `AKL_DB_PASSWORD` / `AKL_DB_API_PASSWORD` / `AKL_DB_PIPELINE_PASSWORD`
These map to real Postgres roles (`akl_pipeline`, `akl_api`) created by
`docker/postgres/init/01_databases.sh`. Rotating requires an `ALTER ROLE ... PASSWORD` against the
running database, not just an env var change:

```bash
docker compose exec postgres psql -U postgres -c "ALTER ROLE akl_api PASSWORD '<new password>';"
```

Update the corresponding `.env`/`.env.prod` value and restart the services using that role
(`akl-api`, and anything running `akl-cli` locally) — the old password stops working the moment
`ALTER ROLE` commits.

## MinIO access/secret key, Qdrant API key
Rotate via each service's own admin path (MinIO console/`mc admin user`, Qdrant's API-key config),
then update `AKL_S3_ACCESS_KEY`/`AKL_S3_SECRET_KEY` or the Qdrant API key setting and restart
every consumer (`akl-api`, Airflow workers, and any host running `akl-cli`).

## `AKL_LLM_API_KEY`
Rotate at the LLM provider, update the env var, restart `akl-api`. No caching of this value
outside process memory.
