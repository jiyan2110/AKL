"""``akl-cli`` entry point."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer

from akl import __version__
from akl.config import Settings
from akl.errors import AKLError

app = typer.Typer(
    name="akl-cli", help="Enterprise AI Knowledge Lakehouse - operator CLI.", no_args_is_help=True
)
config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"akl-cli {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """Root callback."""


def _load_settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        typer.echo(json.dumps(exc.details, indent=2), err=True)
        raise typer.Exit(code=2) from exc


@config_app.command("show")
def config_show(
    config_file: Annotated[Path | None, typer.Option("--config-file", "-c")] = None,
) -> None:
    """Print resolved configuration with secrets redacted."""
    settings = _load_settings(config_file)
    typer.echo(json.dumps(settings.redacted(), indent=2, default=str))


def _probe_tcp(host: str, port: int, timeout: float) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return (
                True,
                f"tcp connect {host}:{port} in {(time.perf_counter() - start) * 1000:.0f} ms",
            )
    except OSError as exc:
        return False, f"tcp connect {host}:{port} failed: {exc}"


def _probe_http(url: str, timeout: float) -> tuple[bool, str]:
    start = time.perf_counter()
    request = urllib.request.Request(  # noqa: S310 - operator-supplied HTTP probe URL
        url, method="GET", headers={"User-Agent": f"akl-cli/{__version__}"}
    )  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            code = int(response.status)
    except urllib.error.HTTPError as exc:
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        return False, f"GET {url} failed: {exc}"
    return (
        200 <= code < 400,
        f"GET {url} -> {code} in {(time.perf_counter() - start) * 1000:.0f} ms",
    )


@config_app.command("check")
def config_check(
    config_file: Annotated[Path | None, typer.Option("--config-file", "-c")] = None,
    timeout: Annotated[float, typer.Option(help="Per-probe timeout in seconds.")] = 3.0,
) -> None:
    """Validate settings and probe PostgreSQL, MinIO and Qdrant."""
    settings = _load_settings(config_file)
    qdrant = urlparse(settings.qdrant.url)
    checks: list[tuple[str, tuple[bool, str]]] = [
        ("postgres", _probe_tcp(settings.db.host, settings.db.port, timeout)),
        ("minio", _probe_http(f"{settings.s3.endpoint.rstrip('/')}/minio/health/live", timeout)),
        ("qdrant-http", _probe_http(f"{settings.qdrant.url.rstrip('/')}/healthz", timeout)),
        (
            "qdrant-grpc",
            _probe_tcp(qdrant.hostname or "localhost", settings.qdrant.grpc_port, timeout),
        ),
    ]
    all_ok = True
    for name, (ok, detail) in checks:
        all_ok &= ok
        typer.secho(
            f"[{'OK ' if ok else 'FAIL'}] {name:<12} {detail}",
            fg=typer.colors.GREEN if ok else typer.colors.RED,
        )
    typer.echo(f"env={settings.core.env.value} config_file={settings.config_file}")
    raise typer.Exit(code=0 if all_ok else 1)


if __name__ == "__main__":
    app()
