# API Reference

The full OpenAPI spec is generated from the running FastAPI app, not maintained by hand:

- **Interactive (Swagger UI)**: `http://localhost:8000/docs` while the API is running (`make api`).
- **Raw spec**: `akl-cli api openapi --out openapi.json` writes it to a file without starting a
  server, or fetch `http://localhost:8000/openapi.json` directly.

## Endpoint groups

| Prefix | Purpose | Auth scope |
|---|---|---|
| `/v1/search` | Hybrid search | `search:read` |
| `/v1/chat` | Streaming/non-streaming cited chat | `chat:write` |
| `/v1/conversations` | Conversation history | `chat:write` (own conversations) |
| `/v1/documents` | Upload, list, detail, chunks, delete | `documents:write` / `documents:delete` |
| `/v1/sources` | Connector status, GitHub sync trigger | `search:read` / `documents:write` |
| `/v1/jobs/{run_id}` | Background job status | `search:read` |
| `/v1/health` | Liveness/readiness/dependencies | none |
| `/v1/admin/documents/{id}/permissions` | Change security level/groups | `documents:permissions` |
| `/v1/admin/api-keys` | API key lifecycle | `keys:manage` |
| `/v1/admin/audit` | Query the audit log | `audit:read` |
| `/v1/admin/gdpr/principals/{id}` | Erase/export a principal's data | self, or `gdpr:manage` |
| `/v1/admin/lineage` | Dataset/run/document lineage | `admin:reload` / `search:read` |
| `/metrics` | Prometheus scrape endpoint | none |

See [error codes](errors.md) for the full failure-mode reference and the response envelope shape.
