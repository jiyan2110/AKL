"""PostgreSQL connectivity and Alembic migration commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config as AlembicConfig

import akl.db as akl_db_pkg
from akl.config import Settings
from akl.db.session import Database
from akl.errors import AKLError

db_app = typer.Typer(
    help="PostgreSQL metadata database: health and migrations.", no_args_is_help=True
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]
MIGRATIONS_DIR = Path(akl_db_pkg.__file__).resolve().parent / "migrations"


def _settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _alembic_config(settings: Settings) -> AlembicConfig:
    config = AlembicConfig()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", Database(settings).url.replace("%", "%%"))
    config.set_main_option("file_template", "%%(rev)s_%%(slug)s")
    return config


def _run(settings: Settings, action: str, fn: Callable[..., object], **kwargs: object) -> None:
    try:
        fn(_alembic_config(settings), **kwargs)
    except Exception as exc:
        typer.secho(f"db {action} failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@db_app.command("ping")
def db_ping(config_file: ConfigOpt = None) -> None:
    settings = _settings(config_file)
    database = Database(settings)
    try:
        result = database.ping()
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        database.dispose()
    typer.secho(
        f"[OK ] postgres  {result.server_version} | user={result.current_user} db={result.database} | {result.latency_ms:.0f} ms",
        fg=typer.colors.GREEN,
    )


@db_app.command("upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="Target revision (default head).")] = "head",
    sql: Annotated[bool, typer.Option("--sql", help="Print SQL instead of executing.")] = False,
    config_file: ConfigOpt = None,
) -> None:
    settings = _settings(config_file)
    _run(settings, "upgrade", command.upgrade, revision=revision, sql=sql)
    if not sql:
        typer.secho(f"[OK ] upgraded to {revision}", fg=typer.colors.GREEN)


@db_app.command("downgrade")
def db_downgrade(
    revision: Annotated[str, typer.Argument(help="Target revision, e.g. '-1' or 'base'.")],
    config_file: ConfigOpt = None,
) -> None:
    settings = _settings(config_file)
    if settings.core.env.value == "prod":
        typer.secho(
            "refusing to downgrade in prod without manual Alembic invocation",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    _run(settings, "downgrade", command.downgrade, revision=revision)
    typer.secho(f"[OK ] downgraded to {revision}", fg=typer.colors.YELLOW)


@db_app.command("current")
def db_current(config_file: ConfigOpt = None) -> None:
    _run(_settings(config_file), "current", command.current, verbose=True)


@db_app.command("history")
def db_history(config_file: ConfigOpt = None) -> None:
    _run(_settings(config_file), "history", command.history, verbose=True, indicate_current=True)
