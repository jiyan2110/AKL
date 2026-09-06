# Runbook: Incident Response

## Triage order

1. **`GET /v1/health/ready`** — the single fastest signal. Check the `dependencies` array and
   `failing` list; this tells you which backend is actually down before you go looking anywhere else.
2. **Grafana → AKL / API Overview** — error rate, latency, in-flight requests. Confirm scope
   (one route vs. everything) and onset time.
3. **Grafana → AKL / Pipelines & Freshness** — is a DAG stale (`akl_dag_stale`) or did a quality
   gate fail (`akl_gate_failures_total`)? A stale ingestion pipeline looks like "search returns
   nothing new," not like an outage.
4. **`docker compose logs -f --tail=200 <service>`** for the specific failing dependency.

## Common scenarios

### 5xx spike on `/v1/search` or `/v1/chat`
- Check `akl_retrieval_degraded_total` — if only one backend (dense/sparse) is down, requests
  should still succeed with a `dense_unavailable`/`sparse_unavailable` flag, not a 5xx. A genuine
  5xx spike with both backends healthy is an application bug — check `akl-api` logs for a
  traceback and the `request_id` from the response body to search the logs precisely.
- If Qdrant is down: `docker compose ps qdrant`; check `docker compose logs qdrant`. Search still
  works via BM25 alone once the retriever notices (may take one request to fail over).
- If Postgres is down: almost everything fails (auth, chunk metadata, conversations). This is the
  most severe single-dependency failure; treat as a full outage.

### `akl_dag_stale{dag_id=...} == 1`
- Check the Airflow UI (`make airflow-up`, `http://localhost:8080`) for the DAG's last run.
- A quality-gate failure (`AKL-E7001`) intentionally blocks the pipeline rather than publishing
  bad data — this is not a bug, it's the gate working. Read the task logs for the actual gate
  breach (quarantine ratio, coverage, drift) before overriding anything.
- To catch up manually without Airflow: `akl-cli pipeline ingestion` / `chunking` / `embedding` /
  `qdrant-sync` run the exact same task entrypoints.

### Qdrant drift (`akl_qdrant_drift != 0`)
- This should be self-healing: the next `akl_qdrant_sync` DAG run (or `make eval-calibrate`-adjacent
  `akl-cli qdrant sync`) reconciles it and raises `AKL-E5020` if it can't. If drift persists across
  multiple syncs, check whether Qdrant lost data (volume issue) — a persistent, non-zero drift after
  a clean sync is a data-loss signal, not a timing issue.

### High refusal rate (`akl_insufficient_evidence_total` climbing)
- Check whether an ingestion/chunking/embedding DAG failed silently upstream (see freshness above)
  — a stale corpus looks exactly like "the model got worse."
- Run `akl-cli eval calibrate` against the current QA set before touching
  `AKL_RAG_MIN_CONFIDENCE` — don't guess a new threshold from a single incident.

## Escalation
There is no paging integration configured by default (`docker/alertmanager/alertmanager.yml`
ships with a log-only receiver — see [Alerting](alerting.md)). If you're reading this because an
alert fired, the receiver you actually use is whatever your team wired into Alertmanager.
