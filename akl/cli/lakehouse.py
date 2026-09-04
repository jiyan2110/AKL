"""Lakehouse storage commands."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import typer

from akl.config import Settings
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.io import LakehouseIO, Layer

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
