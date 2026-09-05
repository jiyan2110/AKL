"""``akl-cli query`` — inspect query processing (PRD §6.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.errors import AKLError
from akl.rag.query import QueryProcessor
from akl.rag.query.filters import to_qdrant_filter
from akl.security.principal import Principal

query_app = typer.Typer(help="Query processing diagnostics.", no_args_is_help=True)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


@query_app.command("inspect")
def inspect(
    text: Annotated[str, typer.Argument(help="Query text.")],
    levels: Annotated[
        str, typer.Option(help="Comma-separated security levels of the principal.")
    ] = "public,internal,restricted",
    groups: Annotated[str, typer.Option(help="Comma-separated groups of the principal.")] = "",
    config_file: ConfigOpt = None,
) -> None:
    """Show normalisation, intent, entities, inferred filters and the compiled Qdrant filter."""
    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    principal = Principal(
        "cli", frozenset(g for g in groups.split(",") if g), frozenset(levels.split(","))
    )
    try:
        processed = QueryProcessor(settings.retrieval).process(text, principal)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(processed.trace(), indent=2, default=str))
    typer.echo(
        json.dumps(
            to_qdrant_filter(principal, processed.hard_filters, processed.soft_filters).model_dump(
                exclude_none=True
            ),
            indent=2,
        )
    )
