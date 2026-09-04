"""Parquet-on-S3 IO for the lakehouse."""

from __future__ import annotations

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
        view = f"akl_write_{run_id.replace('-', '_')}"
        target = self.uri(layer, dataset)
        order_clause = (
            f" ORDER BY {', '.join(_sql_ident(column) for column in sort_by)}" if sort_by else ""
        )
        compression = self._compression_clause()
        metadata = f"KV_METADATA {{ {_sql_str(SCHEMA_VERSION_KEY)}: {_sql_str(schema_version)} }}"
        filename_pattern = f"FILENAME_PATTERN 'part-{run_id}-{{uuid}}'"
        if partition_by:
            destination = target
            part_clause = (
                f"PARTITION_BY ({', '.join(_sql_ident(column) for column in partition_by)})"
            )
            options = f"FORMAT PARQUET, {compression}, {part_clause}, APPEND true, {filename_pattern}, {metadata}"
        else:
            destination = f"{target}/part-{run_id}-{uuid4().hex[:12]}.parquet"
            options = f"FORMAT PARQUET, {compression}, {metadata}"
        self._engine.register(view, table)
        try:
            self._engine.execute(
                f"COPY (SELECT * FROM {view}{order_clause}) TO {_sql_str(destination)} ({options})"  # noqa: S608 - validated SQL fragments
            )
        finally:
            self._engine.unregister(view)
        files = [file for file in self.list_files(layer, dataset) if f"part-{run_id}-" in file.key]
        return WriteResult(
            self.uri(layer, dataset),
            table.num_rows,
            tuple(file.key for file in files),
            sum(file.size_bytes for file in files),
        )

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

    def delete_keys(self, keys: Sequence[str]) -> int:
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

    def _compression_clause(self) -> str:
        lakehouse = self._settings.lakehouse
        if lakehouse.parquet_compression is ParquetCompression.ZSTD:
            return f"COMPRESSION ZSTD, COMPRESSION_LEVEL {lakehouse.parquet_zstd_level}"
        return f"COMPRESSION {lakehouse.parquet_compression.value}"
