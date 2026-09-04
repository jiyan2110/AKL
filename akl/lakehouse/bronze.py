"""Bronze content-addressed raw store and manifest."""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pyarrow as pa

from akl import ids
from akl.errors import AKLError
from akl.lakehouse.io import LakehouseIO, Layer, WriteResult
from akl.lakehouse.schemas import enforce
from akl.lakehouse.schemas.bronze import MANIFEST, SOURCE_TYPES

_EXT_BY_MIME: dict[str, str] = {
    "application/pdf": "pdf",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "txt",
    "text/x-rst": "rst",
    "application/json": "json",
}


class BronzeError(AKLError):
    code = "AKL-E3022"
    http_status = 500
    retryable = True


@dataclass(frozen=True)
class RawPutResult:
    object_key: str
    content_sha256: str
    size_bytes: int
    deduplicated: bool


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension_for(mime_type: str | None, filename: str | None = None) -> str:
    if mime_type:
        clean_mime = mime_type.split(";", 1)[0].strip().lower()
        ext = _EXT_BY_MIME.get(clean_mime) or mimetypes.guess_extension(clean_mime)
        if ext:
            return ext.lstrip(".").lower()
    if filename and "." in filename:
        return filename.rsplit(".", 1)[1].lower()[:10] or "bin"
    return "bin"


def raw_key(source_type: str, content_sha256: str, ext: str) -> str:
    if source_type not in SOURCE_TYPES:
        raise BronzeError(
            f"unknown source_type {source_type!r}",
            details={"allowed": list(SOURCE_TYPES)},
            retryable=False,
        )
    if len(content_sha256) != 64 or any(char not in "0123456789abcdef" for char in content_sha256):
        raise BronzeError("content_sha256 must be 64 lowercase hex chars", retryable=False)
    return f"bronze/raw/source_type={source_type}/sha256={content_sha256}.{ext.strip('.').lower()}"


class BronzeStore:
    def __init__(self, io: LakehouseIO) -> None:
        self._io = io

    def put_raw(
        self,
        data: bytes,
        *,
        source_type: str,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> RawPutResult:
        if not data:
            raise BronzeError("refusing to store empty object", retryable=False)
        sha = sha256_hex(data)
        key = raw_key(source_type, sha, extension_for(mime_type, filename))
        if self._io.object_exists(key):
            return RawPutResult(key, sha, len(data), deduplicated=True)
        self._io.put_object(
            key, data, content_type=mime_type, metadata={"sha256": sha, "source-type": source_type}
        )
        return RawPutResult(key, sha, len(data), deduplicated=False)

    def get_raw(self, object_key: str) -> bytes:
        data = self._io.get_object(object_key)
        expected = object_key.rsplit("sha256=", 1)[-1].split(".", 1)[0]
        actual = sha256_hex(data)
        if actual != expected:
            raise BronzeError(
                "bronze object checksum mismatch", details={"key": object_key, "actual": actual}
            )
        return data

    def raw_exists(self, object_key: str) -> bool:
        return self._io.object_exists(object_key)

    @staticmethod
    def build_manifest_row(
        *,
        source_uri: str,
        source_type: str,
        put: RawPutResult,
        connector_name: str,
        connector_version: str,
        run_id: str,
        mime_type: str | None,
        fetched_at: datetime | None = None,
        source_metadata: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        fetched = (fetched_at or datetime.now(UTC)).astimezone(UTC)
        canonical = ids.canonicalize_uri(source_uri)
        doc_id = ids.document_id(canonical)
        return {
            "manifest_id": str(ids.manifest_id(doc_id, put.content_sha256, run_id)),
            "document_id": str(doc_id),
            "content_sha256": put.content_sha256,
            "source_type": source_type,
            "source_uri": source_uri,
            "canonical_source_uri": canonical,
            "object_key": put.object_key,
            "size_bytes": put.size_bytes,
            "mime_type": mime_type,
            "fetched_at": fetched,
            "connector_name": connector_name,
            "connector_version": connector_version,
            "run_id": run_id,
            "source_metadata": list((source_metadata or {}).items()),
            "ingest_date": fetched.date(),
        }

    def write_manifest(self, rows: Sequence[Mapping[str, Any]], *, run_id: str) -> WriteResult:
        if not rows:
            return WriteResult(self._io.uri(Layer.BRONZE, "manifest"), 0, (), 0)
        table = enforce(pa.Table.from_pylist(list(rows), schema=MANIFEST.schema), MANIFEST)
        return self._io.write(
            table,
            Layer.BRONZE,
            "manifest",
            run_id=run_id,
            schema_version=MANIFEST.version,
            partition_by=MANIFEST.partition_by,
            sort_by=MANIFEST.sort_by,
        )

    def read_manifest(
        self, *, ingest_date: date | None = None, where: str | None = None
    ) -> pa.Table:
        partition = f"ingest_date={ingest_date.isoformat()}" if ingest_date else None
        return self._io.read(Layer.BRONZE, "manifest", partition=partition, where=where)

    def find_by_sha(self, content_sha256: str) -> pa.Table:
        return self._io.read(Layer.BRONZE, "manifest", where=f"content_sha256 = '{content_sha256}'")


def new_run_id(prefix: str = "cli") -> str:
    return f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
