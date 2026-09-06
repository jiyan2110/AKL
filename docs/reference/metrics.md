# Metrics Reference

Generated from the module-level Prometheus metric objects in `akl/observability/metrics.py` (the scraped API registry) by `scripts/generate_docs_reference.py` — do not edit by hand; regenerate with `make docs-reference`. `PipelineMetrics` (pushed per Airflow/CLI task via Pushgateway, one fresh registry per task) is instantiated at runtime and is not listed here; see its class docstring in the same module for that catalog.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `akl_answer_citations` | Histogram | — | Citations per answer |
| `akl_answer_flags` | Counter | flag | Answer flags |
| `akl_auth_failures` | Counter | reason | 401/403 responses |
| `akl_chat_requests` | Counter | mode, stream | Chat requests |
| `akl_dag_last_success_timestamp_seconds` | Gauge | dag_id | Unix time of the last successful run per DAG |
| `akl_dag_stale` | Gauge | dag_id | 1 if the DAG's last success is older than its configured staleness threshold |
| `akl_http_rate_limited` | Counter | route_class | 429 responses |
| `akl_http_request_duration_seconds` | Histogram | method, route | HTTP latency |
| `akl_http_requests` | Counter | method, route, status | HTTP requests |
| `akl_http_requests_inflight` | Gauge | route | In-flight requests |
| `akl_insufficient_evidence` | Counter | reason | Refusals |
| `akl_lineage_edges_written` | Counter | edge_type | Lineage edges recorded |
| `akl_llm_latency_seconds` | Histogram | phase | LLM latency |
| `akl_rerank_confidence` | Histogram | intent | Top rerank score |
| `akl_retrieval_degraded` | Counter | backend | Dense/sparse backend failures seen by the API |
| `akl_search_latency_seconds` | Histogram | stage | Search latency by stage |
| `akl_search_requests` | Counter | mode, intent | Search requests |
| `akl_sse_streams_active` | Gauge | — | Active SSE streams |
| `akl_upload_bytes` | Counter | source_type | Uploaded bytes |
| `akl_uploads` | Counter | source_type, outcome | Uploads |
