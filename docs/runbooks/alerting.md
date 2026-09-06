# Runbook: Alerting

Alert rules live in `docker/prometheus/alerts.yml` (loaded by the `prometheus` service) and are
grouped `akl-api` / `akl-pipelines`. **Alertmanager ships with a log-only receiver by default**
(`docker/alertmanager/alertmanager.yml`) — alerts fire and are visible in Alertmanager's UI and
`docker compose logs alertmanager`, but nothing pages anyone until you configure a real receiver.

## Wiring a real receiver
Replace the `receivers:` block in `docker/alertmanager/alertmanager.yml` with your provider (Slack
webhook, PagerDuty routing key, email SMTP) and restart the `alertmanager` service. Do this before
relying on this stack for anything you'd actually want to be woken up for.

## Current alert rules

| Alert | Severity | Fires when |
|---|---|---|
| `AKLHighErrorRate` | page | 5xx rate > 5% over 5m |
| `AKLSearchLatencyHigh` | warn | p95 search latency > 2s over 10m |
| `AKLInsufficientEvidenceSpike` | warn | refusal rate > 0.5/s over 15m |
| `AKLDagStale` | warn | a DAG hasn't succeeded within its configured freshness threshold |
| `AKLQdrantDrift` | page | Qdrant point count doesn't match Gold after a sync |
| `AKLEmbeddingCoverageLow` | warn | embedding coverage < 99% for 30m |
| `AKLGateFailures` | page | any quality gate failed in the last hour |

`severity: page` is a suggestion for how urgently a human should look, not a guarantee of
actually paging anyone — that depends entirely on the receiver you configure above.
