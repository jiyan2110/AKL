# Error Code Reference

Generated from `code = "AKL-..."` class attributes across the codebase by `scripts/generate_docs_reference.py` — do not edit by hand; regenerate with `make docs-reference`.

| Code | Class | Description | Source |
|---|---|---|---|
| `AKL-E0001` | `ConfigError` | Invalid or missing settings detected at startup. | `akl\errors.py` |
| `AKL-E0002` | `ConnectorConfigError` | Invalid connector YAML (AKL-E0002 family). | `akl\ingestion\connectors\base.py` |
| `AKL-E1001` | `AuthError` | AuthError | `akl\security\auth.py` |
| `AKL-E1002` | `InvalidTokenError` | InvalidTokenError | `akl\security\auth.py` |
| `AKL-E1003` | `ForbiddenError` | ForbiddenError | `akl\security\auth.py` |
| `AKL-E1005` | `InvalidApiKeyError` | InvalidApiKeyError | `akl\security\auth.py` |
| `AKL-E1006` | `RateLimitedError` | RateLimitedError | `akl\api\middleware\ratelimit.py` |
| `AKL-E1007` | `ApiKeyStoreUnavailableError` | No database configured for API-key storage (AKL-E1007) — a service-availability issue, | `akl\security\auth.py` |
| `AKL-E1008` | `AuthConfigError` | The server has no signing secret configured (AKL-E1008) — an operator misconfiguration, | `akl\security\auth.py` |
| `AKL-E2001` | `LakehouseEngineError` | LakehouseEngineError | `akl\lakehouse\engine.py` |
| `AKL-E2002` | `LakehouseIOError` | LakehouseIOError | `akl\lakehouse\io.py` |
| `AKL-E2003` | `DatasetNotFoundError` | DatasetNotFoundError | `akl\lakehouse\io.py` |
| `AKL-E2101` | `SchemaEnforcementError` | SchemaEnforcementError | `akl\lakehouse\schemas\__init__.py` |
| `AKL-E2102` | `CompactionError` | CompactionError | `akl\lakehouse\compaction.py` |
| `AKL-E3001` | `UploadTooLargeError` | UploadTooLargeError | `akl\api\routers\documents.py` |
| `AKL-E3003` | `UnsupportedFormatError` | No parser accepts this MIME/extension (AKL-E3003). | `akl\ingestion\parsers\base.py` |
| `AKL-E3010` | `PdfEncryptedError` | PdfEncryptedError | `akl\ingestion\parsers\pdf.py` |
| `AKL-E3011` | `PdfCorruptError` | PdfCorruptError | `akl\ingestion\parsers\pdf.py` |
| `AKL-E3012` | `PdfTooLargeError` | PdfTooLargeError | `akl\ingestion\parsers\pdf.py` |
| `AKL-E3013` | `PdfNoTextError` | PdfNoTextError | `akl\ingestion\parsers\pdf.py` |
| `AKL-E3020` | `ConnectorError` | Source unreachable or fetch failed after retries (AKL-E3020). | `akl\ingestion\connectors\base.py` |
| `AKL-E3022` | `BronzeError` | BronzeError | `akl\lakehouse\bronze.py` |
| `AKL-E3023` | `DatabaseUnavailableError` | DatabaseUnavailableError | `akl\db\session.py` |
| `AKL-E3030` | `ParseError` | Parser failed on an input (AKL-E3030). | `akl\ingestion\parsers\base.py` |
| `AKL-E3040` | `GitHubRateLimitError` | GitHubRateLimitError | `akl\ingestion\connectors\github.py` |
| `AKL-E3041` | `GitHubNotFoundError` | GitHubNotFoundError | `akl\ingestion\connectors\github.py` |
| `AKL-E3060` | `HardDeleteConfirmationRequiredError` | Missing/wrong ``X-Confirm`` header on a hard delete (AKL-E3060) — a client input error, not | `akl\api\routers\documents.py` |
| `AKL-E5001` | `EmbeddingModelError` | Model download/load failure (AKL-E5001). | `akl\embedding\provider.py` |
| `AKL-E5002` | `EmbeddingError` | Inference failure (AKL-E5002). | `akl\embedding\provider.py` |
| `AKL-E5010` | `QdrantSchemaError` | QdrantSchemaError | `akl\embedding\qdrant\schema.py` |
| `AKL-E5011` | `QdrantUnavailableError` | QdrantUnavailableError | `akl\embedding\qdrant\schema.py` |
| `AKL-E5020` | `QdrantDriftError` | Qdrant point count differs from Gold after sync (AKL-E5020). | `akl\embedding\qdrant\reconciler.py` |
| `AKL-E5030` | `Bm25IndexError` | Index missing or unloadable (AKL-E5030). | `akl\embedding\bm25\index.py` |
| `AKL-E6001` | `EmptyQueryError` | EmptyQueryError | `akl\rag\query\normalize.py` |
| `AKL-E6012` | `RetrievalUnavailableError` | Both retrieval backends failed (AKL-E6012). | `akl\rag\retrieval\engine.py` |
| `AKL-E6021` | `LLMConfigError` | LLMConfigError | `akl\rag\llm\provider.py` |
| `AKL-E6030` | `ConversationNotFoundError` | Conversation missing or owned by another principal (AKL-E6030). | `akl\rag\memory.py` |
| `AKL-E7001` | `GateFailed` | GateFailed | `akl\pipelines\airflow_tasks.py` |
| `AKL-E7002` | `JobNotFoundError` | JobNotFoundError | `akl\api\routers\sources.py` |
| `AKL-E9010` | `LineageNotFoundError` | LineageNotFoundError | `akl\api\routers\admin\lineage.py` |
| `AKL-E9020` | `ApiKeyNotFoundError` | ApiKeyNotFoundError | `akl\api\routers\admin\keys.py` |
| `AKL-E9030` | `GdprUnavailableError` | GdprUnavailableError | `akl\api\routers\admin\gdpr.py` |
| `AKL-W6020` | `LLMUnavailableError` | Generation backend timed out / errored (AKL-W6020 → extractive fallback). | `akl\rag\llm\provider.py` |
