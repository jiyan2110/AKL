"""``akl-cli ingest`` — run connectors, parse the Bronze backlog, inspect quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from akl.config import Settings
from akl.db.models import QuarantineItem
from akl.db.session import Database
from akl.errors import AKLError
from akl.ingestion.service import IngestionService
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine

ingest_app = typer.Typer(
    help="Ingestion: connectors → Bronze → Silver (with validation & quarantine).",
    no_args_is_help=True,
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _service(config_file: Path | None) -> tuple[IngestionService, DuckDBEngine, Database]:
    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    engine = DuckDBEngine(settings)
    db = Database(settings)
    return IngestionService(settings, engine, db), engine, db


def _parse(service: IngestionService, run_id: str, *, limit: int = 500) -> None:
    rep = service.parse_backlog(run_id=run_id, limit=limit)
    colour = typer.colors.GREEN if rep.quarantined == 0 else typer.colors.YELLOW
    typer.secho(
        f"[OK ] parse   considered={rep.considered} parsed={rep.parsed} skipped={rep.skipped} "
        f"quarantined={rep.quarantined} duplicates={rep.duplicates} low_quality={rep.low_quality}",
        fg=colour,
    )
    for f in rep.failures:
        typer.secho(f"       {f['code']} {f['document_id']}: {f['error']}", fg=typer.colors.RED)


@ingest_app.command("connectors")
def list_connectors(config_file: ConfigOpt = None) -> None:
    """List configured connectors and their health."""
    service, engine, db = _service(config_file)
    try:
        for cfg in service.connector_configs():
            health = service.connectors.create(cfg).health()
            colour = typer.colors.GREEN if health.ok else typer.colors.RED
            typer.secho(f"{cfg.id:<24} type={cfg.type:<9} {health.detail}", fg=colour)
    finally:
        engine.close()
        db.dispose()


@ingest_app.command("run")
def run(
    connector: Annotated[
        str | None, typer.Option("--connector", "-k", help="Connector id; omit to run all enabled.")
    ] = None,
    parse: Annotated[
        bool, typer.Option("--parse/--no-parse", help="Parse Bronze backlog afterwards.")
    ] = True,
    config_file: ConfigOpt = None,
) -> None:
    """Discover + fetch to Bronze (recording documents), then parse to Silver."""
    service, engine, db = _service(config_file)
    run_id = new_run_id("cli")
    try:
        ids = [connector] if connector else [c.id for c in service.connector_configs()]
        for cid in ids:
            report = service.run_connector(cid, run_id=run_id)
            typer.secho(
                f"[OK ] fetch   {cid}: discovered={report.discovered} fetched={report.fetched} dedup={report.deduplicated} "
                f"failed={report.failed} deletions={len(report.deletions)} manifest_rows={report.manifest_rows} ({report.duration_s:.1f}s)",
                fg=typer.colors.GREEN if report.failed == 0 else typer.colors.YELLOW,
            )
            for f in report.failures:
                typer.secho(f"       {f['code']} {f['uri']}: {f['error']}", fg=typer.colors.RED)
        if parse:
            _parse(service, run_id)
    except (AKLError, KeyError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
    typer.echo(json.dumps({"run_id": run_id}))


@ingest_app.command("parse")
def parse_only(
    limit: Annotated[int, typer.Option(help="Max documents to parse.")] = 500,
    config_file: ConfigOpt = None,
) -> None:
    """Parse documents in status 'bronze' into Silver."""
    service, engine, db = _service(config_file)
    try:
        _parse(service, new_run_id("cli"), limit=limit)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()


@ingest_app.command("quarantine")
def quarantine(
    status: Annotated[str, typer.Option(help="open | retried | dismissed | all")] = "open",
    limit: Annotated[int, typer.Option()] = 50,
    config_file: ConfigOpt = None,
) -> None:
    """List quarantined items."""
    _service_, engine, db = _service(config_file)
    try:
        with db.session() as s:
            stmt = select(QuarantineItem).order_by(QuarantineItem.created_at.desc()).limit(limit)
            if status != "all":
                stmt = stmt.where(QuarantineItem.status == status)
            rows = list(s.scalars(stmt))
            for q in rows:
                typer.echo(
                    f"{q.created_at:%Y-%m-%d %H:%M}  {q.error_code:<10} {q.stage:<14} {str(q.document_id)[:8]}…  {(q.detail or '')[:80]}"
                )
            typer.echo(f"({len(rows)} item(s), status={status})")
    finally:
        engine.close()
        db.dispose()
