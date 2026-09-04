"""Lakehouse storage commands."""

from __future__ import annotations

import json
import mimetypes
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import typer

from akl.config import Settings
from akl.errors import AKLError
from akl.lakehouse.bronze import BronzeStore, new_run_id
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO, Layer
from akl.lakehouse.silver import SilverStore

lakehouse_app = typer.Typer(
    help="Lakehouse storage operations (DuckDB + MinIO).", no_args_is_help=True
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _fail(exc: AKLError) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
    raise typer.Exit(code=1)


@lakehouse_app.command("init")
def lakehouse_init(config_file: ConfigOpt = None) -> None:
    """Smoke-test bucket access, partitioned IO, reads, listing, and cleanup."""
    settings = _settings(config_file)
    run_id = f"smoke-{uuid.uuid4().hex[:8]}"
    try:
        with DuckDBEngine(settings) as engine:
            io = LakehouseIO(settings, engine)
            io.ensure_bucket()
            typer.secho(f"[OK ] bucket        s3://{io.bucket} reachable", fg=typer.colors.GREEN)
            engine.ensure_s3()
            typer.secho(
                "[OK ] duckdb        httpfs loaded, S3 secret created", fg=typer.colors.GREEN
            )
            table = pa.table(
                {
                    "id": pa.array([1, 2, 3], pa.int64()),
                    "source_type": pa.array(["pdf", "pdf", "markdown"]),
                    "payload": pa.array(["a", "b", "c"]),
                }
            )
            result = io.write(
                table,
                Layer.SCRATCH,
                "smoke",
                run_id=run_id,
                schema_version="0.0.1",
                partition_by=["source_type"],
                sort_by=["id"],
            )
            typer.secho(
                f"[OK ] write         {result.rows} rows -> {len(result.files)} file(s), {result.bytes_written} bytes",
                fg=typer.colors.GREEN,
            )
            full = io.read(Layer.SCRATCH, "smoke", where="payload IN ('a','b','c')")
            pruned = io.read(Layer.SCRATCH, "smoke", partition="source_type=pdf")
            ok = (
                full.num_rows == 3 and pruned.num_rows == 2 and "source_type" in pruned.column_names
            )
            typer.secho(
                f"[{'OK ' if ok else 'FAIL'}] read          full={full.num_rows} pruned(pdf)={pruned.num_rows} hive_cols={'source_type' in pruned.column_names}",
                fg=typer.colors.GREEN if ok else typer.colors.RED,
            )
            files = [
                file.key for file in io.list_files(Layer.SCRATCH, "smoke") if run_id in file.key
            ]
            deleted = io.delete_keys(files)
            typer.secho(
                f"[OK ] cleanup       deleted {deleted} scratch file(s)", fg=typer.colors.GREEN
            )
            if not ok:
                raise typer.Exit(code=1)
            typer.echo("Lakehouse storage initialised and verified.")
    except AKLError as exc:
        _fail(exc)


@lakehouse_app.command("query")
def lakehouse_query(
    sql: Annotated[str, typer.Argument(help="SQL to execute.")],
    config_file: ConfigOpt = None,
    limit: Annotated[int, typer.Option(help="Max rows to print.")] = 50,
    as_json: Annotated[bool, typer.Option("--json", help="Print rows as JSON lines.")] = False,
) -> None:
    """Run ad-hoc SQL through DuckDB."""
    settings = _settings(config_file)
    try:
        with DuckDBEngine(settings) as engine:
            table = engine.execute(sql)
    except AKLError as exc:
        _fail(exc)
        return
    rows = table.slice(0, limit).to_pylist()
    if as_json:
        for row in rows:
            typer.echo(json.dumps(row, default=str))
    elif not rows:
        typer.echo("(0 rows)")
    else:
        columns = table.column_names
        widths = [max(len(column), *(len(str(row[column])) for row in rows)) for column in columns]
        typer.echo(
            " | ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True))
        )
        typer.echo("-+-".join("-" * width for width in widths))
        for row in rows:
            typer.echo(
                " | ".join(
                    str(row[column]).ljust(width)
                    for column, width in zip(columns, widths, strict=True)
                )
            )
    typer.echo(f"({table.num_rows} rows total, showing {len(rows)})")


@lakehouse_app.command("ls")
def lakehouse_ls(
    layer: Annotated[
        str, typer.Argument(help="bronze | silver | gold | quarantine | backups | _scratch")
    ],
    dataset: Annotated[str, typer.Argument(help="Dataset name")],
    config_file: ConfigOpt = None,
) -> None:
    """List Parquet files for a dataset with sizes."""
    settings = _settings(config_file)
    try:
        with DuckDBEngine(settings) as engine:
            io = LakehouseIO(settings, engine)
            files = io.list_files(layer, dataset)
    except AKLError as exc:
        _fail(exc)
        return
    total = 0
    for file in files:
        total += file.size_bytes
        typer.echo(f"{file.size_bytes:>12,}  {file.last_modified:%Y-%m-%d %H:%M:%S}  {file.key}")
    typer.echo(f"{len(files)} file(s), {total:,} bytes under s3://{io.bucket}/{layer}/{dataset}/")


@lakehouse_app.command("bronze-put")
def bronze_put(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    source_type: Annotated[str, typer.Option("--source-type", "-t")],
    uri: Annotated[str | None, typer.Option("--uri")] = None,
    mime_type: Annotated[str | None, typer.Option("--mime")] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Store a file content-addressed in Bronze and append a manifest row."""
    settings = _settings(config_file)
    source_uri = uri or path.resolve().as_uri()
    mime = mime_type or mimetypes.guess_type(path.name)[0]
    run_id = new_run_id("cli")
    try:
        with DuckDBEngine(settings) as engine:
            store = BronzeStore(LakehouseIO(settings, engine))
            put = store.put_raw(
                path.read_bytes(), source_type=source_type, mime_type=mime, filename=path.name
            )
            row = BronzeStore.build_manifest_row(
                source_uri=source_uri,
                source_type=source_type,
                put=put,
                connector_name="cli",
                connector_version="0.1.0",
                run_id=run_id,
                mime_type=mime,
                source_metadata={"filename": path.name},
            )
            result = store.write_manifest([row], run_id=run_id)
    except AKLError as exc:
        _fail(exc)
        return
    typer.secho(
        f"[OK ] raw       {put.object_key} ({put.size_bytes} bytes) deduplicated={put.deduplicated}",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"[OK ] manifest  {result.rows} row -> {result.files[0] if result.files else '?'}",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        json.dumps(
            {
                "document_id": row["document_id"],
                "content_sha256": put.content_sha256,
                "run_id": run_id,
            },
            indent=2,
        )
    )


@lakehouse_app.command("bronze-ls")
def bronze_ls(
    ingest_date: Annotated[str | None, typer.Option("--date")] = None,
    limit: Annotated[int, typer.Option(help="Max rows to print.")] = 50,
    config_file: ConfigOpt = None,
) -> None:
    """List Bronze manifest rows."""
    settings = _settings(config_file)
    try:
        with DuckDBEngine(settings) as engine:
            store = BronzeStore(LakehouseIO(settings, engine))
            table = store.read_manifest(
                ingest_date=date.fromisoformat(ingest_date) if ingest_date else None
            )
    except (AKLError, ValueError) as exc:
        if isinstance(exc, AKLError):
            _fail(exc)
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    columns = [
        "ingest_date",
        "source_type",
        "document_id",
        "content_sha256",
        "size_bytes",
        "source_uri",
        "run_id",
    ]
    for row in table.select(columns).slice(0, limit).to_pylist():
        row["content_sha256"] = str(row["content_sha256"])[:12] + "..."
        row["document_id"] = str(row["document_id"])[:8] + "..."
        typer.echo("  ".join(f"{key}={row[key]}" for key in columns))
    typer.echo(f"({table.num_rows} manifest rows)")


@lakehouse_app.command("silver-status")
def silver_status(config_file: ConfigOpt = None) -> None:
    """Register current-state views and print Silver dataset/file counts."""
    settings = _settings(config_file)
    try:
        with DuckDBEngine(settings) as engine:
            io = LakehouseIO(settings, engine)
            store = SilverStore(io, engine)
            for dataset in ("documents", "chunks", "dedup_ledger"):
                files = io.list_files(Layer.SILVER, dataset)
                typer.echo(
                    f"silver/{dataset:<13} files={len(files):<4} bytes={sum(file.size_bytes for file in files):,}"
                )
            for name, count in store.view_counts().items():
                typer.secho(f"{name:<22} rows={count}", fg=typer.colors.GREEN)
    except AKLError as exc:
        _fail(exc)


@lakehouse_app.command("gold-refresh")
def gold_refresh(
    chunk_quality_min: Annotated[float, typer.Option(help="Minimum chunk quality.")] = 0.30,
    doc_quality_min: Annotated[float, typer.Option(help="Minimum document quality.")] = 0.35,
    config_file: ConfigOpt = None,
) -> None:
    """Project newly-current Silver chunks into Gold retrieval units."""
    settings = _settings(config_file)
    run_id = new_run_id("cli")
    try:
        with DuckDBEngine(settings) as engine:
            store = GoldStore(LakehouseIO(settings, engine), engine)
            result, snapshot = store.refresh_retrieval_units(
                run_id=run_id,
                chunk_quality_min=chunk_quality_min,
                doc_quality_min=doc_quality_min,
            )
    except AKLError as exc:
        _fail(exc)
        return
    typer.secho(
        f"[OK ] gold/retrieval_units +{result.rows} rows in {len(result.files)} file(s); gold_snapshot_id={snapshot}",
        fg=typer.colors.GREEN,
    )


@lakehouse_app.command("gold-status")
def gold_status(config_file: ConfigOpt = None) -> None:
    """Register Gold views and print dataset, backlog, and coverage counts."""
    settings = _settings(config_file)
    try:
        with DuckDBEngine(settings) as engine:
            io = LakehouseIO(settings, engine)
            store = GoldStore(io, engine)
            for dataset in ("retrieval_units", "chunk_embeddings", "eval/qa_pairs", "stats"):
                files = io.list_files(Layer.GOLD, dataset)
                typer.echo(
                    f"gold/{dataset:<17} files={len(files):<4} bytes={sum(file.size_bytes for file in files):,}"
                )
            for name, count in store.view_counts().items():
                typer.secho(f"{name:<22} rows={count}", fg=typer.colors.GREEN)
            typer.echo(
                f"embedding_version={store.embedding_version} backlog={store.embedding_backlog().num_rows} coverage={store.coverage_ratio():.3f}"
            )
    except AKLError as exc:
        _fail(exc)
