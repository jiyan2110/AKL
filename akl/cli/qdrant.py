"""``akl-cli qdrant`` — collection schema, Gold→Qdrant sync, drift check, dense search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.db.session import Database
from akl.embedding.pipeline import EmbeddingPipeline
from akl.embedding.qdrant.reconciler import QdrantReconciler
from akl.embedding.qdrant.schema import make_client
from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine
from akl.rag.query import QueryProcessor
from akl.rag.query.filters import to_qdrant_filter
from akl.security.principal import Principal

qdrant_app = typer.Typer(
    help="Qdrant: ensure collection, reconcile from Gold, inspect drift, dense search.",
    no_args_is_help=True,
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _ctx(
    config_file: Path | None,
) -> tuple[QdrantReconciler, EmbeddingPipeline, DuckDBEngine, Database]:
    try:
        settings = Settings.load(config_file=config_file)
        engine = DuckDBEngine(settings)
        db = Database(settings)
        pipeline = EmbeddingPipeline(settings, engine, db)
        return (
            QdrantReconciler(make_client(settings), settings, engine, pipeline.gold),
            pipeline,
            engine,
            db,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _fail(exc: AKLError) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
    raise typer.Exit(code=1)


@qdrant_app.command("ensure")
def ensure(config_file: ConfigOpt = None) -> None:
    """Create the collection, payload indexes and alias (idempotent)."""
    rec, _p, engine, db = _ctx(config_file)
    try:
        st = rec.schema.ensure()
        typer.secho(
            f"[OK ] collection={st.name} points={st.points} dim={st.dim} alias→{st.alias_target} missing_indexes={list(st.missing_indexes)}",
            fg=typer.colors.GREEN,
        )
    except AKLError as exc:
        _fail(exc)
    finally:
        engine.close()
        db.dispose()


@qdrant_app.command("sync")
def sync(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Compute the diff only.")] = False,
    config_file: ConfigOpt = None,
) -> None:
    """Reconcile Qdrant with Gold (upsert missing/stale points, delete orphans, verify counts)."""
    rec, _p, engine, db = _ctx(config_file)
    run_id = new_run_id("cli")
    try:
        rep = rec.sync(run_id=run_id, dry_run=dry_run)
        typer.secho(
            f"[OK ] qdrant  collection={rep.collection} gold={rep.gold_points} before={rep.qdrant_points_before} "
            f"to_upsert={rep.to_upsert} to_delete={rep.to_delete}"
            + (
                ""
                if dry_run
                else f" upserted={rep.upserted} deleted={rep.deleted} after={rep.qdrant_points_after} drift={rep.drift}"
            ),
            fg=typer.colors.GREEN if rep.drift == 0 else typer.colors.RED,
        )
    except AKLError as exc:
        _fail(exc)
    finally:
        engine.close()
        db.dispose()


@qdrant_app.command("diff")
def diff(config_file: ConfigOpt = None) -> None:
    """Alias for ``sync --dry-run``."""
    sync(dry_run=True, config_file=config_file)


@qdrant_app.command("status")
def status(config_file: ConfigOpt = None) -> None:
    rec, _p, engine, db = _ctx(config_file)
    try:
        st = rec.schema.status()
        typer.echo(
            f"collection={st.name} exists={st.exists} points={st.points} dim={st.dim} alias→{st.alias_target} missing_indexes={list(st.missing_indexes)}"
        )
    except AKLError as exc:
        _fail(exc)
    finally:
        engine.close()
        db.dispose()


@qdrant_app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    k: Annotated[int, typer.Option("--k", help="Results to show.")] = 5,
    config_file: ConfigOpt = None,
) -> None:
    """Dense-only search through the alias (hybrid retrieval arrives in Milestone 26)."""
    rec, pipeline, engine, db = _ctx(config_file)
    try:
        processed = QueryProcessor(pipeline.settings.retrieval).process(query, Principal.dev())
        vec = pipeline.provider.embed_query(processed.dense_text)
        hits = rec.search(
            vec, limit=k, query_filter=to_qdrant_filter(processed.principal, processed.hard_filters)
        )
        typer.echo(f"intent={processed.intent.value} entities={processed.entities.as_dict()}")
        for i, h in enumerate(hits, start=1):
            typer.echo(
                f"{i:>2}. {h['score']:.3f}  {h.get('title') or ''} › {h.get('heading_breadcrumb') or ''}  [{h.get('chunk_type')}]"
            )
            typer.echo(f"      {str(h.get('text', ''))[:160].replace(chr(10), ' ')}")
    except AKLError as exc:
        _fail(exc)
    finally:
        engine.close()
        db.dispose()
