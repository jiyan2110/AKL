"""``akl-cli auth`` — mint development JWTs and API keys (PRD §9.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.db.session import Database
from akl.errors import AKLError
from akl.security.auth import Authenticator

auth_app = typer.Typer(help="Authentication helpers.", no_args_is_help=True)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@auth_app.command("mint-token")
def mint_token(
    user: Annotated[str, typer.Option("--user", "-u")] = "dev",
    groups: Annotated[str, typer.Option(help="Comma-separated groups.")] = "eng",
    levels: Annotated[
        str, typer.Option(help="Comma-separated security levels.")
    ] = "public,internal",
    roles: Annotated[
        str, typer.Option(help="Comma-separated roles: reader,contributor,curator,admin,service")
    ] = "reader",
    ttl: Annotated[int | None, typer.Option(help="Seconds; default AKL_JWT_TTL_S.")] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Print a signed HS256 JWT for local use (requires AKL_JWT_SECRET)."""
    settings = _settings(config_file)
    try:
        token = Authenticator(settings, None).mint_token(
            user,
            groups=_split(groups),
            security_levels=_split(levels),
            roles=_split(roles),
            ttl_s=ttl,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(token)


@auth_app.command("create-key")
def create_key(
    name: Annotated[str, typer.Option("--name", "-n")],
    groups: Annotated[str, typer.Option()] = "",
    levels: Annotated[str, typer.Option()] = "public,internal",
    roles: Annotated[str, typer.Option()] = "reader",
    config_file: ConfigOpt = None,
) -> None:
    """Create an API key (stored hashed); the full key is printed once."""
    settings = _settings(config_file)
    db = Database(settings)
    try:
        minted = Authenticator(settings, db).create_api_key(
            name=name, groups=_split(groups), security_levels=_split(levels), roles=_split(roles)
        )
        typer.secho(f"X-API-Key: {minted.token}", fg=typer.colors.GREEN)
        typer.echo(
            f"key_id={minted.key_id} prefix={minted.prefix} (store the key now; it is not retrievable)"
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        db.dispose()


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
