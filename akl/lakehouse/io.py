"""Parquet-on-S3 IO for the lakehouse."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import boto3
import pyarrow as pa
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from akl.config import ParquetCompression, Settings
from akl.errors import AKLError
from akl.lakehouse.engine import QueryEngine

SCHEMA_VERSION_KEY = "akl.schema_version"


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    QUARANTINE = "quarantine"
    BACKUPS = "backups"
    SCRATCH = "_scratch"


class LakehouseIOError(AKLError):
    code = "AKL-E2002"
    http_status = 500
    retryable = True


class DatasetNotFoundError(AKLError):
    code = "AKL-E2003"
    http_status = 500
    retryable = False


@dataclass(frozen=True)
class FileInfo:
    key: str
    size_bytes: int
    last_modified: datetime


@dataclass(frozen=True)
class WriteResult:
    uri: str
    rows: int
    files: tuple[str, ...]
    bytes_written: int


def run_token(run_id: str) -> str:
    """Deterministic object-key-safe form of any run id (Airflow ids contain ``: + .``).

    Ids that are already ``[A-Za-z0-9_-]`` are returned unchanged (CLI/API ids, existing keys stay
    identical). Anything else is mapped to ``_`` and gets a short hash suffix so distinct ids never
    collapse onto the same token.
    """
    safe = re.sub(r"[^0-9A-Za-z_-]", "_", run_id)
    if safe == run_id:
        return run_id
    digest = hashlib.blake2b(run_id.encode("utf-8"), digest_size=4).hexdigest()
    return f"{safe[:48]}_{digest}"


def run_ident(run_id: str) -> str:
    """SQL-identifier-safe form (``[A-Za-z0-9_]`` only) for DuckDB view names."""
    return run_token(run_id).replace("-", "_")


def dataset_uri(bucket: str, layer: Layer | str, dataset: str) -> str:
    layer_str = layer.value if isinstance(layer, Layer) else layer
    return f"s3://{bucket}/{layer_str.strip('/')}/{dataset.strip('/')}"


def dataset_prefix(layer: Layer | str, dataset: str) -> str:
    layer_str = layer.value if isinstance(layer, Layer) else layer
    return f"{layer_str.strip('/')}/{dataset.strip('/')}/"


def partition_path(**partitions: Any) -> str:
    return "/".join(f"{key}={value}" for key, value in partitions.items())


def _sql_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier: {name!r}")
    return f'"{name}"'


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LakehouseIO:
    def __init__(self, settings: Settings, engine: QueryEngine) -> None:
        self._settings = settings
        self._engine = engine
        self._bucket = settings.s3.bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3.endpoint,
            region_name=settings.s3.region,
            aws_access_key_id=settings.s3.access_key.get_secret_value(),
            aws_secret_access_key=settings.s3.secret_key.get_secret_value(),
            use_ssl=settings.s3.use_ssl,
            config=BotoConfig(
                s3={"addressing_style": "path" if settings.s3.path_style else "virtual"},
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    def uri(self, layer: Layer | str, dataset: str) -> str:
        return dataset_uri(self._bucket, layer, dataset)

    def ensure_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError(
                f"bucket '{self._bucket}' not accessible",
                details={"endpoint": self._settings.s3.endpoint, "error": str(exc)},
            ) from exc

    def write(
        self,
        table: pa.Table,
        layer: Layer | str,
        dataset: str,
        *,
        run_id: str,
        schema_version: str,
        partition_by: Sequence[str] = (),
        sort_by: Sequence[str] = (),
    ) -> WriteResult:
        if table.num_rows == 0:
            return WriteResult(self.uri(layer, dataset), 0, (), 0)
        for column in (*partition_by, *sort_by):
            if column not in table.column_names:
                raise LakehouseIOError(
                    f"column '{column}' not in table", details={"columns": table.column_names}
                )
        token = run_token(run_id)
        view = f"akl_write_{run_ident(run_id)}"
        target = self.uri(layer, dataset)
        order_clause = (
            f" ORDER BY {', '.join(_sql_ident(column) for column in sort_by)}" if sort_by else ""
        )
        compression = self._compression_clause()
        metadata = f"KV_METADATA {{ {_sql_str(SCHEMA_VERSION_KEY)}: {_sql_str(schema_version)} }}"
        filename_pattern = f"FILENAME_PATTERN 'part-{token}-{{uuid}}'"
        if partition_by:
            destination = target
            part_clause = (
                f"PARTITION_BY ({', '.join(_sql_ident(column) for column in partition_by)})"
            )
            options = f"FORMAT PARQUET, {compression}, {part_clause}, APPEND true, {filename_pattern}, {metadata}"
        else:
            destination = f"{target}/part-{token}-{uuid4().hex[:12]}.parquet"
            options = f"FORMAT PARQUET, {compression}, {metadata}"
        self._engine.register(view, table)
        try:
            self._engine.execute(
                f"COPY (SELECT * FROM {view}{order_clause}) TO {_sql_str(destination)} ({options})"  # noqa: S608 - validated SQL fragments
            )
        finally:
            self._engine.unregister(view)
        files = [file for file in self.list_files(layer, dataset) if f"part-{token}-" in file.key]
        return WriteResult(
            self.uri(layer, dataset),
            table.num_rows,
            tuple(file.key for file in files),
            sum(file.size_bytes for file in files),
        )

    def write_file(
        self, table: pa.Table, layer: Layer | str, dataset: str, *, partition: str, run_id: str
    ) -> str:
        """Write ``table`` as ONE Parquet file inside ``<dataset>/<partition>/`` (compaction target)."""
        token = run_token(run_id)
        view = f"akl_writefile_{run_ident(run_id)}"
        base = self.uri(layer, dataset)
        target_dir = f"{base}/{partition.strip('/')}" if partition else base
        destination = f"{target_dir}/part-{token}-{uuid4().hex[:12]}.parquet"
        self._engine.register(view, table)
        try:
            self._engine.execute(
                f"COPY (SELECT * FROM {view}) TO {_sql_str(destination)} "
                f"(FORMAT PARQUET, {self._compression_clause()})"
            )
        finally:
            self._engine.unregister(view)
        return destination.split(f"s3://{self._bucket}/", 1)[1]

    def read(
        self,
        layer: Layer | str,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        where: str | None = None,
        partition: str | None = None,
    ) -> pa.Table:
        if not self.list_files(layer, dataset, partition=partition):
            raise DatasetNotFoundError(
                f"no files under {self.uri(layer, dataset)}/{partition or ''}"
            )
        selected = ", ".join(_sql_ident(column) for column in columns) if columns else "*"
        sql = f"SELECT {selected} FROM {self.read_expression(layer, dataset, partition=partition)}"  # noqa: S608 - internal SQL fragments
        if where:
            sql += f" WHERE {where}"
        return self._engine.execute(sql)

    def read_expression(
        self, layer: Layer | str, dataset: str, *, partition: str | None = None
    ) -> str:
        base = self.uri(layer, dataset)
        glob = f"{base}/{partition.strip('/')}/*.parquet" if partition else f"{base}/**/*.parquet"
        return f"read_parquet({_sql_str(glob)}, hive_partitioning = true, union_by_name = true)"

    def list_files(
        self, layer: Layer | str, dataset: str, *, partition: str | None = None
    ) -> list[FileInfo]:
        prefix = dataset_prefix(layer, dataset)
        if partition:
            prefix += partition.strip("/") + "/"
        files: list[FileInfo] = []
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".parquet"):
                        files.append(FileInfo(obj["Key"], int(obj["Size"]), obj["LastModified"]))
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError(
                "list failed", details={"prefix": prefix, "error": str(exc)}
            ) from exc
        return files

    def delete_keys(self, keys: Sequence[str], *, allow_bronze_raw: bool = False) -> int:
        forbidden = [key for key in keys if key.startswith("bronze/raw/")]
        if forbidden:
            raise LakehouseIOError(
                "refusing to delete immutable bronze/raw objects", details={"keys": forbidden[:5]}
            )
        deleted = 0
        try:
            for index in range(0, len(keys), 1000):
                batch = [{"Key": key} for key in keys[index : index + 1000]]
                if batch:
                    self._s3.delete_objects(
                        Bucket=self._bucket, Delete={"Objects": batch, "Quiet": True}
                    )
                    deleted += len(batch)
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError("delete failed", details={"error": str(exc)}) from exc
        return deleted

    def list_objects(self, prefix: str) -> list[FileInfo]:
        """All objects under ``prefix`` with size and last-modified (any extension)."""
        out: list[FileInfo] = []
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    out.append(FileInfo(obj["Key"], int(obj["Size"]), obj["LastModified"]))
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError(
                "list failed", details={"prefix": prefix, "error": str(exc)}
            ) from exc
        return out

    def list_keys(self, prefix: str) -> list[str]:
        """All object keys under ``prefix`` (any extension)."""
        out: list[str] = []
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                out.extend(obj["Key"] for obj in page.get("Contents", []))
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError(
                "list failed", details={"prefix": prefix, "error": str(exc)}
            ) from exc
        return out

    def object_exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise LakehouseIOError("head failed", details={"key": key, "error": str(exc)}) from exc
        except BotoCoreError as exc:
            raise LakehouseIOError("head failed", details={"key": key, "error": str(exc)}) from exc

    def put_object(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Upload bytes, refusing to overwrite immutable Bronze raw objects."""
        if key.startswith("bronze/raw/") and self.object_exists(key):
            raise LakehouseIOError(
                "refusing to overwrite immutable bronze/raw object", details={"key": key}
            )
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        if metadata:
            kwargs["Metadata"] = metadata
        try:
            self._s3.put_object(**kwargs)
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError("put failed", details={"key": key, "error": str(exc)}) from exc

    def get_object(self, key: str) -> bytes:
        try:
            body = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"]
            return bytes(body.read())
        except (ClientError, BotoCoreError) as exc:
            raise LakehouseIOError("get failed", details={"key": key, "error": str(exc)}) from exc

    def _compression_clause(self) -> str:
        lakehouse = self._settings.lakehouse
        if lakehouse.parquet_compression is ParquetCompression.ZSTD:
            return f"COMPRESSION ZSTD, COMPRESSION_LEVEL {lakehouse.parquet_zstd_level}"
        return f"COMPRESSION {lakehouse.parquet_compression.value}"
