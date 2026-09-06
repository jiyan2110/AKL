# Runbook: Backup and Restore

## What gets backed up

The `akl_maintenance` DAG (or `akl-cli pipeline maintenance`) runs two backup steps nightly:

- **`backup_postgres`** — `pg_dump -Fc` of the `akl` database, written to
  `backups/postgres/<db>-<timestamp>.dump` in MinIO. Skipped (not failed) if `pg_dump` isn't on
  the task's PATH — check the report for `"skipped": true` before assuming it ran.
- **`qdrant_snapshot`** — a native Qdrant collection snapshot, copied to
  `backups/qdrant/<collection>/<name>` in MinIO.

Both are subject to `backup_retention` (default 14 days — `configs/dags/maintenance.yaml`).

**Not backed up separately**: Bronze/Silver/Gold Parquet in MinIO. These are your source of
truth for content; back up the MinIO bucket itself with your object-storage provider's own
snapshot/replication mechanism, not through this pipeline.

## Restoring Postgres

```bash
# Pull the dump out of MinIO (mc, aws s3, or the MinIO console), then:
docker compose exec -T postgres pg_restore --clean --if-exists -U akl_pipeline -d akl < akl-<timestamp>.dump
```

Restore into a **fresh** database when possible and cut over, rather than restoring in place —
`pg_restore --clean` will drop and recreate objects, which is destructive if anything is still
writing to the target.

## Restoring Qdrant

```bash
# Upload the snapshot file back to the running Qdrant instance, then:
curl -X PUT "http://localhost:6333/collections/<collection>/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///qdrant/snapshots/<name>"}'
```

After a Qdrant restore, run `akl-cli qdrant sync --dry-run` first — if Gold and the restored
snapshot disagree (e.g. the snapshot predates recent embeddings), you'll see it in the diff before
committing to a live sync.

## Full disaster recovery order

1. Restore Postgres (metadata: documents, chunks, conversations, audit log).
2. Restore the MinIO bucket (Bronze/Silver/Gold Parquet) from your object-storage provider's backup.
3. Restore Qdrant (or skip and let `akl-cli qdrant sync` rebuild it from Gold + the restored
   `chunk_embeddings` Parquet — slower, but doesn't need a Qdrant-specific backup at all).
4. Rebuild BM25: `akl-cli bm25 build` (fast; it's a Gold projection, never worth backing up
   separately).
5. `make test-component` against the restored stack before reopening to traffic.
