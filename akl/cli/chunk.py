"""``akl-cli chunk`` — run the chunking engine and inspect chunk statistics."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from akl.chunking.incremental import ChunkingService
from akl.config import Settings
from akl.db.repositories.chunks import ChunkRepository
from akl.db.repositories.runs import RunRepository
from akl.db.session import Database
from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine

chunk_app = typer.Typer(
    help="Chunking: Silver documents → Silver chunks → Gold retrieval units.", no_args_is_help=True
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _service(config_file: Path | None) -> tuple[ChunkingService, DuckDBEngine, Database]:
    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    engine = DuckDBEngine(settings)
    db = Database(settings)
    return ChunkingService(settings, engine, db), engine, db


@chunk_app.command("run")
def run(
    document_id: Annotated[
        list[str] | None, typer.Option("--document-id", "-d", help="Restrict to document id(s).")
    ] = None,
    limit: Annotated[int, typer.Option(help="Max documents per run.")] = 200,
    gold: Annotated[
        bool, typer.Option("--gold/--no-gold", help="Refresh gold/retrieval_units afterwards.")
    ] = True,
    config_file: ConfigOpt = None,
) -> None:
    """Chunk documents lacking chunks for the active chunker version/config (incremental)."""
    service, engine, db = _service(config_file)
    run_id = new_run_id("cli")

    with db.session() as s:
        RunRepository(s).start_run(run_id, "akl_chunking")

    state = "failed"
    rep = None

    try:
        ids = [uuid.UUID(d) for d in document_id] if document_id else None
        rep = service.run(run_id=run_id, document_ids=ids, limit=limit, refresh_gold=gold)
        state = "success"

    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    finally:
        with db.session() as s:
            RunRepository(s).finish_run(
                run_id,
                state=state,
                gold_snapshot_id=getattr(rep, "gold_snapshot_id", None) if rep else None,
            )

        engine.close()
        db.dispose()

    colour = typer.colors.GREEN if rep.documents_failed == 0 else typer.colors.YELLOW
    typer.secho(
        f"[OK ] chunk   documents considered={rep.documents_considered} chunked={rep.documents_chunked} "
        f"unchanged={rep.documents_unchanged} failed={rep.documents_failed} | chunks written={rep.chunks_written} "
        f"tombstoned={rep.chunks_tombstoned} (unchanged={rep.unchanged} modified={rep.modified} moved={rep.moved} "
        f"added={rep.added} removed={rep.removed} reparented={rep.reparented})",
        fg=colour,
    )

    for f in rep.failures:
        typer.secho(f"       {f['code']} {f['document_id']}: {f['error']}", fg=typer.colors.RED)

    if gold:
        typer.secho(
            f"[OK ] gold    retrieval_units +{rep.gold_rows_promoted} rows; gold_snapshot_id={rep.gold_snapshot_id}",
            fg=typer.colors.GREEN,
        )

    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "chunker_version": service.settings.chunking.chunker_version,
                "chunk_config_hash": service.chunker.config_hash,
            }
        )
    )


@chunk_app.command("stats")
def stats(config_file: ConfigOpt = None) -> None:
    """Chunk counts in Postgres and the current-state views, token/quality distribution."""
    service, engine, db = _service(config_file)
    try:
        with db.session() as s:
            counts = ChunkRepository(s).counts()
        typer.echo(
            f"postgres chunks: total={counts['total']} current={counts['current']} embedding_pending={counts['embedding_pending']}"
        )
        views = service.gold.view_counts()
        for name, n in views.items():
            typer.secho(f"{name:<22} rows={n}", fg=typer.colors.GREEN)
        table = service.silver.current_chunks()
        if table.num_rows:
            rows = engine.execute(
                "SELECT chunk_type, count(*) AS n, round(avg(token_count),1) AS avg_tokens, round(avg(quality_score),3) AS avg_quality "
                "FROM v_current_chunks GROUP BY 1 ORDER BY 1"
            ).to_pylist()
            for r in rows:
                typer.echo(
                    f"  {r['chunk_type']:<8} n={r['n']:<6} avg_tokens={r['avg_tokens']:<7} avg_quality={r['avg_quality']}"
                )
        typer.echo(
            f"tokenizer={service.chunker.counter.backend} chunker_version={service.settings.chunking.chunker_version} config_hash={service.chunker.config_hash}"
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
