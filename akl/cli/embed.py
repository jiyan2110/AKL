"""``akl-cli embed`` — embedding backlog, generation and coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.db.repositories.embedding_cache import EmbeddingCacheRepository
from akl.db.session import Database
from akl.embedding.pipeline import EmbeddingPipeline
from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine

embed_app = typer.Typer(
    help="Embeddings: Gold backlog → cache → bge-small → gold/chunk_embeddings.",
    no_args_is_help=True,
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _pipeline(config_file: Path | None) -> tuple[EmbeddingPipeline, DuckDBEngine, Database]:
    try:
        settings = Settings.load(config_file=config_file)
        engine = DuckDBEngine(settings)
        db = Database(settings)
        return EmbeddingPipeline(settings, engine, db), engine, db
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=2) from exc


@embed_app.command("run")
def run(
    limit: Annotated[int | None, typer.Option(help="Max chunks to embed this run.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override AKL_EMBED_BATCH_SIZE.")] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Embed every active Gold unit lacking a vector for the configured embedding version."""
    pipeline, engine, db = _pipeline(config_file)
    run_id = new_run_id("cli")
    try:
        rep = pipeline.run(run_id=run_id, limit=limit, batch_size=batch_size)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
    colour = typer.colors.GREEN if rep.failed == 0 else typer.colors.YELLOW
    typer.secho(
        f"[OK ] embed   version={rep.embedding_version} backlog={rep.backlog} cache_hits={rep.cache_hits} "
        f"generated={rep.generated} written={rep.written} failed={rep.failed} batches={rep.batches} "
        f"({rep.duration_s:.1f}s, {rep.throughput_cps} chunks/s)",
        fg=colour,
    )
    for f in rep.failures:
        typer.secho(f"       batch {f['batch']}: {f['error']}", fg=typer.colors.RED)
    typer.echo(json.dumps({"run_id": run_id, "job_id": str(rep.job_id)}))


@embed_app.command("status")
def status(config_file: ConfigOpt = None) -> None:
    """Coverage ratio, backlog size, cache size and provider details."""
    pipeline, engine, db = _pipeline(config_file)
    try:
        ratio, backlog = pipeline.coverage()
        with db.session() as s:
            cached = EmbeddingCacheRepository(s).count()
        typer.echo(
            f"provider={pipeline.settings.embedding.embed_provider} model={pipeline.provider.model_id} dim={pipeline.provider.dim}"
        )
        typer.echo(f"embedding_version={pipeline.embedding_version}")
        typer.secho(
            f"coverage={ratio:.3f} backlog={backlog} cache_rows={cached}",
            fg=typer.colors.GREEN if backlog == 0 else typer.colors.YELLOW,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()


@embed_app.command("warm")
def warm(config_file: ConfigOpt = None) -> None:
    """Download/load the embedding model and embed a probe sentence."""
    pipeline, engine, db = _pipeline(config_file)
    try:
        vec = pipeline.provider.embed_query("warm up")
        typer.secho(
            f"[OK ] model {pipeline.provider.model_id} loaded; dim={len(vec)} norm={float((vec**2).sum()) ** 0.5:.4f}",
            fg=typer.colors.GREEN,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
