"""Qdrant collection schema, payload indexes and alias management (PRD §5.8, Appendix D)."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from akl.config import Settings
from akl.errors import AKLError

PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "document_id": qm.PayloadSchemaType.KEYWORD,
    "source_type": qm.PayloadSchemaType.KEYWORD,
    "security_level": qm.PayloadSchemaType.KEYWORD,
    "allowed_groups": qm.PayloadSchemaType.KEYWORD,
    "chunk_type": qm.PayloadSchemaType.KEYWORD,
    "code_language": qm.PayloadSchemaType.KEYWORD,
    "repo": qm.PayloadSchemaType.KEYWORD,
    "embedding_version": qm.PayloadSchemaType.KEYWORD,
    "token_count": qm.PayloadSchemaType.INTEGER,
    "document_updated_at": qm.PayloadSchemaType.INTEGER,
    "quality_score": qm.PayloadSchemaType.FLOAT,
    "untrusted": qm.PayloadSchemaType.BOOL,
}

# Payload contract (PRD §2.6.1) — every key copied from gold/retrieval_units into the point payload.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "chunk_key",
    "lineage_id",
    "chunk_checksum",
    "embedded_text_sha256",
    "document_id",
    "document_version_id",
    "chunk_index",
    "source_type",
    "canonical_source_uri",
    "source_uri",
    "title",
    "heading_path",
    "heading_breadcrumb",
    "chunk_type",
    "code_language",
    "text",
    "context_prefix",
    "token_count",
    "page_start",
    "page_end",
    "line_start",
    "line_end",
    "security_level",
    "allowed_groups",
    "repo",
    "branch",
    "path",
    "document_updated_at",
    "quality_score",
    "quality_flags",
    "language",
    "gold_snapshot_id",
)


class QdrantUnavailableError(AKLError):
    code = "AKL-E5011"
    http_status = 503
    retryable = True


class QdrantSchemaError(AKLError):
    code = "AKL-E5010"
    http_status = 503
    retryable = False


@dataclass(frozen=True)
class CollectionStatus:
    name: str
    exists: bool
    points: int
    dim: int | None
    alias_target: str | None
    missing_indexes: tuple[str, ...]


def make_client(settings: Settings, *, timeout: float = 30.0) -> QdrantClient:
    q = settings.qdrant
    try:
        return QdrantClient(
            url=q.url,
            grpc_port=q.grpc_port,
            prefer_grpc=q.prefer_grpc,
            api_key=q.api_key.get_secret_value() if q.api_key else None,
            timeout=int(timeout),
        )
    except Exception as exc:  # pragma: no cover - construction rarely fails
        raise QdrantUnavailableError(
            "cannot create Qdrant client", details={"url": q.url, "error": str(exc)}
        ) from exc


class QdrantSchema:
    """Idempotent collection/index/alias management."""

    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        *,
        dim: int,
        collection: str | None = None,
        manage_alias: bool = True,
    ) -> None:
        self.client = client
        self.manage_alias = manage_alias
        self.settings = settings
        self.dim = dim
        self.collection = collection or settings.qdrant.collection
        self.alias = settings.qdrant.collection_alias

    def ensure(self) -> CollectionStatus:
        """Create the collection (HNSW m=16, ef_construct=128, cosine), payload indexes and alias."""
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
                    hnsw_config=qm.HnswConfigDiff(
                        m=self.settings.qdrant.hnsw_m,
                        ef_construct=self.settings.qdrant.hnsw_ef_construct,
                        full_scan_threshold=10_000,
                    ),
                    optimizers_config=qm.OptimizersConfigDiff(
                        default_segment_number=2, indexing_threshold=20_000
                    ),
                    on_disk_payload=self.settings.qdrant.on_disk_payload,
                )
            info = self.client.get_collection(self.collection)
            size = _vector_size(info)
            if size is not None and size != self.dim:
                raise QdrantSchemaError(
                    f"collection {self.collection} has dimension {size}, expected {self.dim} (AKL-E5012)",
                    details={"collection": self.collection},
                )
            existing = set((info.payload_schema or {}).keys())
            for field, schema in PAYLOAD_INDEXES.items():
                if field not in existing:
                    self.client.create_payload_index(
                        self.collection, field_name=field, field_schema=schema, wait=True
                    )
            if self.manage_alias:
                self.ensure_alias(self.collection)
        except AKLError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(
                "qdrant schema operation failed", details={"error": str(exc)}
            ) from exc
        return self.status()

    def ensure_alias(self, target: str) -> None:
        current = self.alias_target()
        if current == target:
            return
        actions: list[qm.AliasOperations] = []
        if current is not None:
            actions.append(
                qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=self.alias))
            )
        actions.append(
            qm.CreateAliasOperation(
                create_alias=qm.CreateAlias(collection_name=target, alias_name=self.alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=actions)

    def alias_target(self) -> str | None:
        try:
            for a in self.client.get_aliases().aliases:
                if a.alias_name == self.alias:
                    return str(a.collection_name)
        except Exception as exc:
            raise QdrantUnavailableError(
                "cannot read aliases", details={"error": str(exc)}
            ) from exc
        return None

    def status(self) -> CollectionStatus:
        try:
            exists = self.client.collection_exists(self.collection)
            if not exists:
                return CollectionStatus(
                    self.collection, False, 0, None, self.alias_target(), tuple(PAYLOAD_INDEXES)
                )
            info = self.client.get_collection(self.collection)
            existing = set((info.payload_schema or {}).keys())
            count = self.client.count(self.collection, exact=True).count
        except AKLError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError("qdrant unavailable", details={"error": str(exc)}) from exc
        return CollectionStatus(
            self.collection,
            True,
            int(count),
            _vector_size(info),
            self.alias_target(),
            tuple(f for f in PAYLOAD_INDEXES if f not in existing),
        )


def _vector_size(info: qm.CollectionInfo) -> int | None:
    vectors = info.config.params.vectors
    if isinstance(vectors, qm.VectorParams):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        return int(first.size)
    return None
