"""``akl-cli api`` — run the FastAPI gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.errors import AKLError

api_app = typer.Typer(help="FastAPI gateway.", no_args_is_help=True)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


@api_app.command("serve")
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Auto-reload on code changes (dev).")
    ] = False,
    config_file: ConfigOpt = None,
) -> None:
    """Start uvicorn with the app factory (single worker; models are loaded once at startup)."""
    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    import uvicorn

    uvicorn.run(
        "akl.api.main:get_app",
        factory=True,
        host=host or settings.api.api_host,
        port=port or settings.api.api_port,
        reload=reload,
        log_level=settings.core.log_level.value.lower(),
    )


@api_app.command("openapi")
def openapi(
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("openapi.json"),
    config_file: ConfigOpt = None,
) -> None:
    """Write the OpenAPI document to a file without starting a server."""
    import json

    from akl.api.main import create_app

    settings = Settings.load(config_file=config_file)
    app = create_app(settings, warm=False)
    out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    typer.secho(f"[OK ] wrote {out} ({len(app.routes)} routes)", fg=typer.colors.GREEN)
