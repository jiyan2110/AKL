"""``akl-cli pipeline`` — run the DAG task entrypoints without Airflow (PRD §7, `make pipeline`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.pipelines import airflow_tasks as t

pipeline_app = typer.Typer(
    help="Run pipeline stages exactly as the Airflow DAGs do (same task entrypoints).",
    no_args_is_help=True,
)


def _echo(stage: str, payload: dict[str, Any]) -> None:
    trimmed = {k: v for k, v in payload.items() if k != "failures"}
    typer.secho(
        f"[OK ] {stage:<26} {json.dumps(trimmed, default=str)[:300]}", fg=typer.colors.GREEN
    )


def _fail(stage: str, exc: AKLError) -> None:
    typer.secho(f"[ERR] {stage}: {exc}", fg=typer.colors.RED, err=True)
    typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
    raise typer.Exit(code=1)


@pipeline_app.command("ingestion")
def ingestion(
    connectors: Annotated[
        str | None, typer.Option(help="Comma-separated connector ids (default: all enabled).")
    ] = None,
) -> None:
    run_id = new_run_id("cli")
    try:
        ids = t.list_connectors(run_id, [c for c in (connectors or "").split(",") if c] or None)
        reports = [t.fetch_connector(run_id, cid, map_index=i) for i, cid in enumerate(ids)]
        for r in reports:
            _echo(f"fetch:{r['connector_id']}", r)
        parsed = t.parse_backlog(run_id)
        _echo("parse_validate_to_silver", parsed)
        _echo("quality_gate", t.ingestion_gate(reports, parsed))
        t.finish_run(run_id, "akl_ingestion")
    except AKLError as exc:
        t.finish_run(run_id, "akl_ingestion", state="failed")
        _fail("ingestion", exc)


@pipeline_app.command("chunking")
def chunking() -> None:
    run_id = new_run_id("cli")
    try:
        rep = t.chunk_run(run_id)
        _echo("chunk_documents", rep)
        _echo("quality_gate", t.chunking_gate(rep))
        t.finish_run(run_id, "akl_chunking", gold_snapshot_id=rep.get("gold_snapshot_id"))
    except AKLError as exc:
        t.finish_run(run_id, "akl_chunking", state="failed")
        _fail("chunking", exc)


@pipeline_app.command("embedding")
def embedding() -> None:
    run_id = new_run_id("cli")
    try:
        _echo("warm_model_check", t.warm_model(run_id))
        rep = t.embed_run(run_id)
        _echo("embed_backlog", rep)
        _echo("coverage_gate", t.coverage_gate(rep))
        t.finish_run(run_id, "akl_embedding")
    except AKLError as exc:
        t.finish_run(run_id, "akl_embedding", state="failed")
        _fail("embedding", exc)


@pipeline_app.command("qdrant-sync")
def qdrant_sync(
    api_url: Annotated[
        str | None, typer.Option(help="API base URL for the reload notification.")
    ] = None,
) -> None:
    run_id = new_run_id("cli")
    try:
        _echo("qdrant_health_sensor", t.qdrant_health(run_id))
        rep = t.qdrant_sync(run_id)
        _echo("reconcile", rep)
        _echo("rebuild_bm25_index", t.bm25_build(run_id))
        _echo("notify_api_reload", t.notify_api_reload(run_id, api_url=api_url))
        t.finish_run(run_id, "akl_qdrant_sync")
    except AKLError as exc:
        t.finish_run(run_id, "akl_qdrant_sync", state="failed")
        _fail("qdrant-sync", exc)


@pipeline_app.command("maintenance")
def maintenance(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan deletions/compaction without executing.")
    ] = False,
    skip: Annotated[
        str,
        typer.Option(
            help="Comma-separated operations to skip (e.g. backup_postgres,qdrant_snapshot)."
        ),
    ] = "",
) -> None:
    run_id = new_run_id("cli")
    skipped = {s for s in skip.split(",") if s}
    ops: list[tuple[str, dict[str, Any]]] = [
        ("compact_partitions", {"dry_run": dry_run}),
        ("bronze_retention", {"dry_run": dry_run}),
        ("quarantine_retention", {"dry_run": dry_run}),
        ("embedding_cache_eviction", {}),
        ("conversation_ttl", {}),
        ("retire_old_embedding_versions", {"dry_run": dry_run}),
        ("compute_corpus_stats", {}),
        ("backup_postgres", {}),
        ("qdrant_snapshot", {}),
        ("backup_retention", {"dry_run": dry_run}),
        ("vacuum_analyze", {}),
    ]
    failed = False
    for op, kw in ops:
        if op in skipped:
            typer.secho(f"[SKP] {op}", fg=typer.colors.YELLOW)
            continue
        try:
            _echo(op, t.maintenance_task(run_id, op, **kw))
        except AKLError as exc:
            failed = True
            typer.secho(f"[ERR] {op}: {exc}", fg=typer.colors.RED, err=True)
        except Exception as exc:  # backups depend on external tooling; keep going
            failed = True
            typer.secho(f"[ERR] {op}: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
    t.finish_run(run_id, "akl_maintenance", state="failed" if failed else "success")
    if failed:
        raise typer.Exit(code=1)


@pipeline_app.command("run-all")
def run_all(api_url: Annotated[str | None, typer.Option()] = None) -> None:
    """ingestion → chunking → embedding → qdrant-sync (maintenance is separate)."""
    ingestion()
    chunking()
    embedding()
    qdrant_sync(api_url=api_url)


@pipeline_app.command("freshness")
def freshness(
    config_file: Annotated[Path | None, typer.Option("--config-file", "-c")] = None,
) -> None:
    """Show how long since each DAG last succeeded, per configs/settings.yaml thresholds."""
    from akl.config import Settings
    from akl.db.session import Database
    from akl.observability.freshness import check_freshness

    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    db = Database(settings)
    try:
        for f in check_freshness(db, settings.observability):
            age = f"{f.age_minutes:.1f}m ago" if f.age_minutes is not None else "never"
            colour = typer.colors.RED if f.stale else typer.colors.GREEN
            typer.secho(
                f"{f.dag_id:<18} last_success={age:<14} stale_after={f.stale_after_minutes}m  {'STALE' if f.stale else 'ok'}",
                fg=colour,
            )
    finally:
        db.dispose()
