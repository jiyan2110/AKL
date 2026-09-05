"""``akl-cli search`` / ``akl-cli ask`` / ``akl-cli bm25`` — hybrid retrieval and extractive answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from akl.config import Settings
from akl.db.session import Database
from akl.embedding.bm25.builder import build_bm25
from akl.embedding.bm25.index import INDEX_PREFIX, LATEST_KEY, Bm25Index
from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine
from akl.lakehouse.gold import GoldStore
from akl.lakehouse.io import LakehouseIO
from akl.rag.query.filters import MetadataFilters
from akl.rag.service import RAGService
from akl.security.principal import Principal

ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]
bm25_app = typer.Typer(help="BM25 sparse index: build from Gold, inspect.", no_args_is_help=True)


def _settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _principal(levels: str, groups: str) -> Principal:
    return Principal(
        "cli", frozenset(g for g in groups.split(",") if g), frozenset(levels.split(","))
    )


def _filters(
    source_type: str | None, repo: str | None, chunk_type: str | None
) -> MetadataFilters | None:
    f = MetadataFilters(
        source_types=[source_type] if source_type else [],
        repos=[repo] if repo else [],
        chunk_types=[chunk_type] if chunk_type else [],
    )
    return None if f.is_empty() else f


def _gold(settings: Settings, io: LakehouseIO, engine: DuckDBEngine) -> GoldStore:
    return GoldStore(
        io,
        engine,
        embedding_version=settings.embedding.embedding_version,
        view_params={"chunker_version": settings.chunking.chunker_version},
    )


def _explain_sparse(svc: RAGService, io: LakehouseIO, settings: Settings) -> None:
    """When the sparse branch is missing, say exactly why (works with or without service-side diagnostics)."""
    if svc.bm25 is not None:
        return
    reason = getattr(svc, "sparse_error", None)
    if reason:
        typer.secho(f"sparse index not loaded: {reason}", fg=typer.colors.YELLOW)
        return
    try:
        Bm25Index.load(io, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b)
        typer.secho(
            "sparse index loads standalone but RAGService did not receive it — check RAGService(use_sparse=True)",
            fg=typer.colors.YELLOW,
        )
    except AKLError as exc:
        typer.secho(f"sparse index not loaded: {exc}", fg=typer.colors.YELLOW)
        typer.echo(json.dumps(exc.details, indent=2, default=str))
    except Exception as exc:  # corrupt artefact or unexpected error
        typer.secho(f"sparse index not loaded: {type(exc).__name__}: {exc}", fg=typer.colors.YELLOW)
    _print_artifact_status(io)


def _print_artifact_status(io: LakehouseIO) -> None:
    latest = (
        io.get_object(LATEST_KEY).decode("utf-8").strip() if io.object_exists(LATEST_KEY) else None
    )
    typer.echo(f"  {LATEST_KEY} = {latest!r}")
    keys = [k for k in _list(io, f"{INDEX_PREFIX}/version=") if k.endswith("/meta.json")]
    versions = sorted(k.split("version=", 1)[1].split("/", 1)[0] for k in keys)
    typer.echo(f"  available versions ({len(versions)}): {versions[-5:]}")


def _list(io: LakehouseIO, prefix: str) -> list[str]:
    lister = getattr(io, "list_keys", None)
    if lister is not None:
        return list(lister(prefix))
    # fallback for a LakehouseIO without list_keys(): use the raw client
    client = getattr(io, "_s3", None)
    if client is None:
        return []
    out: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=io.bucket, Prefix=prefix):
        out.extend(obj["Key"] for obj in page.get("Contents", []))
    return out


def search_command(
    query: Annotated[str, typer.Argument(help="Query text.")],
    mode: Annotated[str, typer.Option(help="hybrid | dense | sparse")] = "hybrid",
    k: Annotated[int, typer.Option("--k")] = 8,
    source_type: Annotated[str | None, typer.Option()] = None,
    repo: Annotated[str | None, typer.Option()] = None,
    chunk_type: Annotated[str | None, typer.Option()] = None,
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
    levels: Annotated[str, typer.Option()] = "public,internal,restricted",
    groups: Annotated[str, typer.Option()] = "",
    as_json: Annotated[bool, typer.Option("--json")] = False,
    config_file: ConfigOpt = None,
) -> None:
    """Hybrid search (dense ∥ BM25 → RRF → rerank) with security filtering."""
    settings = _settings(config_file)
    engine, db = DuckDBEngine(settings), Database(settings)
    try:
        svc = RAGService(settings, engine, db)
        res = svc.search(
            query,
            _principal(levels, groups),
            mode=mode,
            k=k,
            filters=_filters(source_type, repo, chunk_type),
            rerank=not no_rerank,
        )
        io = LakehouseIO(settings, engine)
        sparse_note: list[str] = []
        if svc.bm25 is None:
            import contextlib
            import io as _io

            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                _explain_sparse(svc, io, settings)
            sparse_note = [line for line in buf.getvalue().splitlines() if line]
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()

    r = res.retrieval
    all_flags = [*svc.flags, *r.flags]
    errors: list[str] = list(getattr(r, "errors", []) or [])
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "results": res.results,
                    "query": res.query.trace(),
                    "retrieval": r.trace(),
                    "flags": all_flags,
                    "errors": errors,
                    "sparse_diagnostics": sparse_note,
                    "timings_ms": res.timings_ms,
                },
                indent=2,
                default=str,
            )
        )
        return
    typer.echo(
        f"intent={res.query.intent.value} confidence={r.confidence} sufficient={r.sufficient} "
        f"reranker={r.reranker} flags={all_flags}"
    )
    typer.echo(
        f"dense={len(r.dense_ids)} sparse={len(r.sparse_ids)} fused={len(r.fused_ids)} timings_ms={res.timings_ms}"
    )
    for line in sparse_note:
        typer.secho(line, fg=typer.colors.YELLOW)
    for err in errors:
        typer.secho(f"retrieval error: {err}", fg=typer.colors.YELLOW)
    for item in res.results:
        s = item["scores"]
        rerank = s["rerank"] if s["rerank"] is not None else "-"
        typer.echo(
            f"{item['rank']:>2}. rerank={rerank} rrf={s['rrf']:.4f} dense={s['dense']} sparse={s['sparse']}  "
            f"{item['title']} › {item['heading_breadcrumb']} [{item['chunk_type']}]"
        )
        typer.echo(f"      {item['locator']}")
        typer.echo(f"      {str(item.get('text', ''))[:160].replace(chr(10), ' ')}")


def ask_command(
    question: Annotated[str, typer.Argument(help="Question.")],
    levels: Annotated[str, typer.Option()] = "public,internal,restricted",
    groups: Annotated[str, typer.Option()] = "",
    as_json: Annotated[bool, typer.Option("--json")] = False,
    config_file: ConfigOpt = None,
) -> None:
    """Answer with citations (extractive mode until the LLM provider lands)."""
    settings = _settings(config_file)
    engine, db = DuckDBEngine(settings), Database(settings)
    try:
        svc = RAGService(settings, engine, db)
        res = svc.answer(question, _principal(levels, groups))
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "mode": res.mode,
                    "answer": res.answer,
                    "citations": res.citations,
                    "confidence": res.confidence,
                    "reason": res.reason,
                    "flags": [*svc.flags, *res.flags],
                    "timings_ms": res.timings_ms,
                },
                indent=2,
                default=str,
            )
        )
        return
    if res.answer is None:
        typer.secho(
            f"insufficient evidence (confidence={res.confidence}, reason={res.reason})",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)
    typer.echo(res.answer)
    typer.echo("")
    for c in res.citations:
        typer.secho(f"[{c['index']}] {c['title']} — {c['locator']}", fg=typer.colors.CYAN)
    typer.echo(
        f"mode={res.mode} confidence={res.confidence} flags={[*svc.flags, *res.flags]} timings_ms={res.timings_ms}"
    )


@bm25_app.command("build")
def bm25_build(config_file: ConfigOpt = None) -> None:
    """Snapshot active Gold units into a new BM25 index version and mark it LATEST."""
    settings = _settings(config_file)
    engine, db = DuckDBEngine(settings), Database(settings)
    try:
        io = LakehouseIO(settings, engine)
        rep = build_bm25(settings, io, _gold(settings, io, engine), version=new_run_id("bm25"))
        typer.secho(
            f"[OK ] bm25    version={rep.version} documents={rep.documents} terms={rep.terms} → {rep.prefix}",
            fg=typer.colors.GREEN,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()


@bm25_app.command("status")
def bm25_status(config_file: ConfigOpt = None) -> None:
    """Show the LATEST pointer, available versions and load the current index."""
    settings = _settings(config_file)
    engine = DuckDBEngine(settings)
    try:
        io = LakehouseIO(settings, engine)
        _print_artifact_status(io)
        version = Bm25Index.latest_version(io)
        if version is None:
            typer.secho("no BM25 index yet — run `akl-cli bm25 build`", fg=typer.colors.YELLOW)
            return
        index = Bm25Index.load(io, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b)
        meta: dict[str, Any] = {
            **index.meta(),
            "load_seconds": round(getattr(index, "load_seconds", 0.0), 3),
        }
        typer.echo(json.dumps(meta, indent=2))
        sample = index.search("chunk_key identity", k=3, exact_terms=["chunk_key"])
        typer.echo(f"probe 'chunk_key identity' → {[h.payload.get('title') for h in sample]}")
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
