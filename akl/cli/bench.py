"""``akl-cli bench`` — latency benchmark against the real RAGService, in-process (PRD §11.7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.db.session import Database
from akl.errors import AKLError
from akl.lakehouse.engine import DuckDBEngine
from akl.security.principal import Principal

bench_app = typer.Typer(help="In-process latency benchmark for search/chat.", no_args_is_help=True)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


@bench_app.command("run")
def run(
    repeats: Annotated[int, typer.Option(help="Times to repeat the query set.")] = 5,
    include_answer: Annotated[
        bool,
        typer.Option(
            "--include-answer/--no-include-answer",
            help="Also benchmark answer() (LLM or extractive).",
        ),
    ] = False,
    max_p95_ms: Annotated[
        float | None, typer.Option(help="Fail (exit 1) if search p95 latency exceeds this.")
    ] = None,
    out_json: Annotated[
        Path | None, typer.Option("--out-json", help="Write the full JSON report here.")
    ] = None,
    out_md: Annotated[
        Path | None,
        typer.Option(
            "--out-md",
            help="Write a markdown report here (default: docs/benchmarks/<timestamp>.md).",
        ),
    ] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Run the default query set through search() (and optionally answer()), report p50/p95/p99."""
    try:
        settings = Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    engine = DuckDBEngine(settings)
    db = Database(settings)
    try:
        from akl.eval.benchmark import run_benchmark
        from akl.rag.service import RAGService

        rag = RAGService(settings, engine, db)
        thresholds = {"p95_ms": max_p95_ms} if max_p95_ms is not None else {}
        report = run_benchmark(
            rag,
            repeats=repeats,
            principal=Principal.dev(),
            include_answer=include_answer,
            thresholds=thresholds,
        )

        colour = typer.colors.GREEN if report.passed else typer.colors.RED
        typer.secho(
            f"[{'OK' if report.passed else 'FAIL'}] bench   n={report.search.n} p50={report.search.p50_ms:.1f}ms p95={report.search.p95_ms:.1f}ms p99={report.search.p99_ms:.1f}ms",
            fg=colour,
        )
        if report.answer:
            typer.echo(
                f"  answer  n={report.answer.n} p50={report.answer.p50_ms:.1f}ms p95={report.answer.p95_ms:.1f}ms p99={report.answer.p99_ms:.1f}ms"
            )
        for failure in report.failures:
            typer.secho(f"  threshold miss: {failure}", fg=typer.colors.RED)

        md_path = out_md or Path("docs/benchmarks") / f"{report.generated_at.replace(':', '-')}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        typer.echo(f"wrote {md_path}")
        if out_json:
            out_json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
            typer.echo(f"wrote {out_json}")
        if not report.passed:
            raise typer.Exit(code=1)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
