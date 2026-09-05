"""Typed application settings with environment and YAML precedence."""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, get_args

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from akl.errors import ConfigError, ConfigFileError

DEFAULT_CONFIG_FILE = Path("configs/settings.yaml")
DEFAULT_ENV_FILE = Path(".env")


class Environment(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SSLMode(StrEnum):
    DISABLE = "disable"
    PREFER = "prefer"
    REQUIRE = "require"
    VERIFY_FULL = "verify-full"


def _is_secret_field(annotation: Any) -> bool:
    return annotation is SecretStr or SecretStr in get_args(annotation)


class _SectionSettings(BaseSettings):
    yaml_section: ClassVar[str] = ""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return env_settings, dotenv_settings, init_settings, file_secret_settings

    @model_validator(mode="before")
    @classmethod
    def _resolve_file_secrets(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prefix = str(cls.model_config.get("env_prefix", ""))
        for name, field in cls.model_fields.items():
            if not _is_secret_field(field.annotation):
                continue
            file_var = f"{prefix}{name}_FILE".upper()
            path = os.environ.get(file_var)
            if not path:
                continue
            try:
                data[name] = Path(path).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ConfigError(
                    f"cannot read secret file for {file_var}",
                    details={"variable": file_var, "path": path, "error": str(exc)},
                ) from exc
        return data


class CoreSettings(_SectionSettings):
    yaml_section: ClassVar[str] = "core"
    model_config = SettingsConfigDict(env_prefix="AKL_", extra="ignore", env_file_encoding="utf-8")

    env: Environment = Environment.DEV
    service_name: str = "akl-api"
    log_level: LogLevel = LogLevel.INFO
    log_sample_debug: bool = False
    log_queries: bool = False
    org_name: str = "Example Org"
    config_dir: Path = Path("configs")
    models_dir: Path = Path("/models")
    tmp_dir: Path = Path("/tmp/akl")  # noqa: S108 - container scratch dir per PRD


class DatabaseSettings(_SectionSettings):
    yaml_section: ClassVar[str] = "database"
    model_config = SettingsConfigDict(
        env_prefix="AKL_DB_", extra="ignore", env_file_encoding="utf-8"
    )

    host: str = "postgres"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "akl"
    user: str = "akl_pipeline"
    password: SecretStr
    pool_min: int = Field(default=5, ge=1)
    pool_max: int = Field(default=20, ge=1)
    sslmode: SSLMode = SSLMode.DISABLE

    @model_validator(mode="after")
    def _check_pool(self) -> DatabaseSettings:
        if self.pool_max < self.pool_min:
            raise ValueError("pool_max must be >= pool_min")
        return self

    def dsn(self, driver: str = "postgresql", *, reveal: bool = False) -> str:
        password = self.password.get_secret_value() if reveal else "***"
        return f"{driver}://{self.user}:{password}@{self.host}:{self.port}/{self.name}?sslmode={self.sslmode.value}"


class S3Settings(_SectionSettings):
    yaml_section: ClassVar[str] = "s3"
    model_config = SettingsConfigDict(
        env_prefix="AKL_S3_", extra="ignore", env_file_encoding="utf-8"
    )

    endpoint: str = "http://minio:9000"
    region: str = "us-east-1"
    bucket: str = "akl-lakehouse"
    access_key: SecretStr
    secret_key: SecretStr
    use_ssl: bool = False
    path_style: bool = True

    @model_validator(mode="after")
    def _check_endpoint(self) -> S3Settings:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must start with http:// or https://")
        return self


class QdrantSettings(_SectionSettings):
    yaml_section: ClassVar[str] = "qdrant"
    model_config = SettingsConfigDict(
        env_prefix="AKL_QDRANT_", extra="ignore", env_file_encoding="utf-8"
    )

    url: str = "http://qdrant:6333"
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = True
    api_key: SecretStr | None = None
    collection_alias: str = "kb_chunks"
    collection: str = "kb_chunks_v1"
    hnsw_m: int = Field(default=16, ge=4)
    hnsw_ef_construct: int = Field(default=128, ge=8)
    on_disk_payload: bool = True
    upsert_batch: int = Field(default=512, ge=1)
    delete_batch: int = Field(default=1000, ge=1)
    scroll_page: int = Field(default=10000, ge=100)


class ParquetCompression(StrEnum):
    ZSTD = "ZSTD"
    SNAPPY = "SNAPPY"
    UNCOMPRESSED = "UNCOMPRESSED"


class LakehouseSettings(_SectionSettings):
    yaml_section: ClassVar[str] = "lakehouse"
    model_config = SettingsConfigDict(env_prefix="AKL_")

    parquet_compression: ParquetCompression = ParquetCompression.ZSTD
    parquet_zstd_level: int = Field(default=3, ge=1, le=22)
    duckdb_memory_limit: str = "4GB"
    duckdb_threads: int = Field(default=4, ge=1)
    lakehouse_use_file_manifest: bool = False


class ChunkingSettings(_SectionSettings):
    """Chunking engine parameters (PRD §4.1, Appendix B.6). Env names match the PRD exactly."""

    yaml_section: ClassVar[str] = "chunking"
    model_config = SettingsConfigDict(env_prefix="AKL_")

    chunker_version: str = "1.0.0"
    chunk_target_tokens: int = Field(default=320, ge=32)
    chunk_max_tokens: int = Field(default=448, ge=64)
    chunk_min_tokens: int = Field(default=64, ge=1)
    chunk_overlap_tokens: int = Field(default=48, ge=0)
    chunk_semantic_enabled: bool = True
    chunk_semantic_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    chunk_code_max_tokens: int = Field(default=400, ge=64)
    chunk_table_max_tokens: int = Field(default=400, ge=64)
    chunk_context_prefix_tokens: int = Field(default=40, ge=0)
    chunk_emit_section_parents: bool = False
    chunk_quality_min: float = Field(default=0.30, ge=0.0, le=1.0)
    doc_quality_min: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> ChunkingSettings:
        if not (self.chunk_min_tokens < self.chunk_target_tokens <= self.chunk_max_tokens):
            raise ValueError("require chunk_min_tokens < chunk_target_tokens <= chunk_max_tokens")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be < chunk_target_tokens")
        return self


class EmbeddingSettings(_SectionSettings):
    """Embedding model and batching (PRD §5, Appendix B.5). Env names match the PRD."""

    yaml_section: ClassVar[str] = "embedding"
    model_config = SettingsConfigDict(env_prefix="AKL_")

    embed_provider: str = "bge"  # bge (ONNX) | hash (deterministic, offline/testing)
    embed_model_id: str = "BAAI/bge-small-en-v1.5"
    embed_model_version: str = "1.5"
    embed_model_sha256: str | None = None
    embed_dim: int = Field(default=384, ge=8)
    embed_device: str = "auto"
    embed_batch_size: int = Field(default=64, ge=1)
    embed_threads: int = Field(default=0, ge=0)  # 0 = onnxruntime default
    embed_onnx_int8: bool = False
    embed_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embed_task_shards: int = Field(default=4, ge=1)
    embedding_cache_ttl_days: int = Field(default=180, ge=1)
    embedding_retire_days: int = Field(default=30, ge=1)
    embedder_version: str = "1.0.0"

    @property
    def embedding_version(self) -> str:
        slug = self.embed_model_id.split("/")[-1]
        return f"{slug}__{self.embed_model_version}__{self.embed_dim}"


class RetrievalSettings(_SectionSettings):
    """Query processing and retrieval parameters (PRD §6, Appendix B.8)."""

    yaml_section: ClassVar[str] = "retrieval"
    model_config = SettingsConfigDict(env_prefix="AKL_")

    query_max_chars: int = Field(default=2000, ge=1)
    query_spell_dual: bool = True
    retrieval_dense_k: int = Field(default=50, ge=1)
    retrieval_sparse_k: int = Field(default=50, ge=1)
    retrieval_fused_k: int = Field(default=40, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rag_top_k: int = Field(default=8, ge=1)
    rag_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    rag_min_candidates: int = Field(default=2, ge=1)
    rag_strong_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    qdrant_hnsw_ef: int = Field(default=128, ge=8)
    rerank_enabled: bool = True
    rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_provider: str = "onnx"  # onnx | lexical | none
    rerank_top_n: int = Field(default=40, ge=1)
    rerank_onnx_int8: bool = False
    rag_context_tokens: int = Field(default=3000, ge=200)
    rag_neighbor_expansion: int = Field(default=0, ge=0, le=2)
    rag_dedupe_jaccard: float = Field(default=0.85, ge=0.0, le=1.0)
    rag_soft_filter_penalty: float = Field(default=0.85, ge=0.0, le=1.0)
    rag_marginal_penalty: float = Field(default=0.9, ge=0.0, le=1.0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)


_SECTIONS: tuple[type[_SectionSettings], ...] = (
    CoreSettings,
    DatabaseSettings,
    S3Settings,
    QdrantSettings,
    LakehouseSettings,
    ChunkingSettings,
    EmbeddingSettings,
    RetrievalSettings,
)


class Settings(BaseModel):
    model_config = {"frozen": True}

    core: CoreSettings
    db: DatabaseSettings
    s3: S3Settings
    qdrant: QdrantSettings
    lakehouse: LakehouseSettings
    chunking: ChunkingSettings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    config_file: Path | None = None

    @model_validator(mode="after")
    def _cross_section_rules(self) -> Settings:
        if self.core.env is Environment.PROD and self.db.sslmode is SSLMode.DISABLE:
            raise ValueError("AKL_DB_SSLMODE must not be 'disable' when AKL_ENV=prod")
        return self

    @classmethod
    def load(
        cls, config_file: Path | None = None, env_file: Path | None = DEFAULT_ENV_FILE
    ) -> Settings:
        resolved_file = config_file or Path(os.environ.get("AKL_CONFIG_FILE", DEFAULT_CONFIG_FILE))
        yaml_data = _read_yaml(resolved_file, required=config_file is not None)
        problems: list[dict[str, Any]] = []
        sections: dict[str, _SectionSettings] = {}
        for section_cls in _SECTIONS:
            init_values = yaml_data.get(section_cls.yaml_section) or {}
            if not isinstance(init_values, dict):
                raise ConfigFileError(
                    f"section '{section_cls.yaml_section}' must be a mapping",
                    details={"file": str(resolved_file)},
                )
            try:
                sections[section_cls.yaml_section] = section_cls(
                    _env_file=str(env_file) if env_file else None,
                    **init_values,
                )
            except ValidationError as exc:
                prefix = str(section_cls.model_config.get("env_prefix", ""))
                for error in exc.errors():
                    field = ".".join(str(part) for part in error["loc"]) or "<section>"
                    problems.append(
                        {
                            "variable": f"{prefix}{field}".upper(),
                            "section": section_cls.yaml_section,
                            "error": error["msg"],
                        }
                    )
        if problems:
            raise ConfigError(
                f"{len(problems)} invalid or missing setting(s)",
                details={"problems": problems, "config_file": str(resolved_file)},
            )
        try:
            return cls(
                core=sections["core"],
                db=sections["database"],
                s3=sections["s3"],
                qdrant=sections["qdrant"],
                lakehouse=sections["lakehouse"],
                chunking=sections["chunking"],
                embedding=sections["embedding"],
                retrieval=sections["retrieval"],
                config_file=resolved_file if resolved_file.exists() else None,
            )
        except ValidationError as exc:
            raise ConfigError(
                "cross-section validation failed",
                details={"problems": [error["msg"] for error in exc.errors()]},
            ) from exc

    def redacted(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigFileError(f"config file not found: {path}", details={"file": str(path)})
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise ConfigFileError(
            f"cannot parse config file: {path}", details={"file": str(path), "error": str(exc)}
        ) from exc
    if not isinstance(loaded, dict):
        raise ConfigFileError("config file root must be a mapping", details={"file": str(path)})
    return loaded


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
