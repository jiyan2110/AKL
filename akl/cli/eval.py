"""``akl-cli eval`` — synthetic QA generation, eval runs, and confidence calibration (PRD Chapter 12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from akl.config import Settings
from akl.db.session import Database
from akl.errors import AKLError
from akl.lakehouse.bronze import new_run_id
from akl.lakehouse.engine import DuckDBEngine
from akl.security.principal import Principal

eval_app = typer.Typer(
    help="Synthetic QA generation and retrieval/answer evaluation.", no_args_is_help=True
)
ConfigOpt = Annotated[Path | None, typer.Option("--config-file", "-c", help="YAML settings file.")]


def _settings(config_file: Path | None) -> Settings:
    try:
        return Settings.load(config_file=config_file)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@eval_app.command("generate-qa")
def generate_qa(
    n: Annotated[
        int, typer.Option(help="Number of QA pairs to generate (including distractors).")
    ] = 50,
    distractor_ratio: Annotated[
        float, typer.Option(help="Fraction of n that are unanswerable distractor questions.")
    ] = 0.15,
    method: Annotated[
        str, typer.Option(help="template | llm (falls back to template per-chunk on failure)")
    ] = "template",
    seed: Annotated[int, typer.Option(help="Random seed for reproducible sampling.")] = 0,
    version: Annotated[
        str | None, typer.Option(help="Version tag; default is a fresh run id.")
    ] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Sample chunks from Gold, generate one question per chunk plus distractors, write to gold/eval/qa_pairs."""
    settings = _settings(config_file)
    engine = DuckDBEngine(settings)
    try:
        from akl.eval.generate_qa import generate_qa_pairs
        from akl.lakehouse.gold import GoldStore
        from akl.lakehouse.io import LakehouseIO
        from akl.rag.llm.provider import build_llm

        io = LakehouseIO(settings, engine)
        gold = GoldStore(
            io,
            engine,
            embedding_version=settings.embedding.embedding_version,
            view_params={"chunker_version": settings.chunking.chunker_version},
        )
        rows = gold.active_units().to_pylist()
        if not rows:
            typer.secho("no active Gold units found; run `make seed` first", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        run_version = version or new_run_id("eval")
        llm = build_llm(settings.llm) if method == "llm" else None
        pairs = generate_qa_pairs(
            rows,
            version=run_version,
            n=n,
            distractor_ratio=distractor_ratio,
            method=method,
            llm=llm,
            seed=seed,
        )
        gold.write_qa_pairs([p.as_row() for p in pairs], run_id=run_version)
        by_method: dict[str, int] = {}
        for p in pairs:
            by_method[p.generation_method] = by_method.get(p.generation_method, 0) + 1
        typer.secho(
            f"[OK ] eval    version={run_version} pairs={len(pairs)} by_method={by_method}",
            fg=typer.colors.GREEN,
        )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()


@eval_app.command("run")
def run(
    version: Annotated[
        str | None, typer.Option(help="QA set version; default is the latest generated.")
    ] = None,
    k: Annotated[int, typer.Option(help="Top-k for recall/precision/nDCG.")] = 10,
    mode: Annotated[str, typer.Option(help="hybrid | dense | sparse")] = "hybrid",
    check_answers: Annotated[
        bool,
        typer.Option(
            "--check-answers/--no-check-answers", help="Also call answer() and score faithfulness."
        ),
    ] = False,
    min_recall: Annotated[
        float | None, typer.Option(help="Fail (exit 1) if recall_at_k falls below this.")
    ] = None,
    min_mrr: Annotated[float | None, typer.Option(help="Fail if MRR falls below this.")] = None,
    min_refusal_precision: Annotated[
        float | None, typer.Option(help="Fail if refusal_precision falls below this.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the full JSON report to this path.")
    ] = None,
    mlflow_run_name: Annotated[
        str | None,
        typer.Option(
            help="Log this eval run to MLflow under this name (needs AKL_MLFLOW_ENABLED=true)."
        ),
    ] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Run every QA pair through real hybrid retrieval (and optionally generation), report metrics."""
    settings = _settings(config_file)
    engine = DuckDBEngine(settings)
    db = Database(settings)
    try:
        from akl.eval.runner import run_eval
        from akl.lakehouse.gold import GoldStore
        from akl.lakehouse.io import LakehouseIO
        from akl.observability.mlflow_tracking import log_metrics, log_params, mlflow_run
        from akl.rag.service import RAGService

        io = LakehouseIO(settings, engine)
        gold = GoldStore(
            io,
            engine,
            embedding_version=settings.embedding.embedding_version,
            view_params={"chunker_version": settings.chunking.chunker_version},
        )
        qa_version = version or gold.latest_qa_version()
        if not qa_version:
            typer.secho(
                "no QA set found; run `akl-cli eval generate-qa` first", fg=typer.colors.YELLOW
            )
            raise typer.Exit(code=1)
        qa_pairs = gold.read_qa_pairs(version=qa_version).to_pylist()
        if not qa_pairs:
            typer.secho(f"QA version {qa_version!r} has no rows", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        rag = RAGService(settings, engine, db)
        thresholds: dict[str, float] = {}
        if min_recall is not None:
            thresholds[f"recall_at_{k}"] = min_recall
        if min_mrr is not None:
            thresholds["mrr"] = min_mrr
        if min_refusal_precision is not None:
            thresholds["refusal_precision"] = min_refusal_precision

        report = run_eval(
            rag,
            qa_pairs,
            principal=Principal.dev(),
            k=k,
            mode=mode,
            check_answers=check_answers,
            thresholds=thresholds,
        )
        colour = typer.colors.GREEN if report.passed else typer.colors.RED
        typer.secho(
            f"[{'OK' if report.passed else 'FAIL'}] eval run  version={qa_version} n={report.aggregate['n']} ({report.duration_s:.1f}s)",
            fg=colour,
        )
        for key, value in report.aggregate.items():
            if key not in ("n", "n_answerable", "n_distractor"):
                typer.echo(
                    f"  {key:<20} {value:.4f}"
                    if isinstance(value, float)
                    else f"  {key:<20} {value}"
                )
        if report.faithfulness_mean is not None:
            typer.echo(f"  {'faithfulness_mean':<20} {report.faithfulness_mean:.4f}")
        for failure in report.failures:
            typer.secho(f"  threshold miss: {failure}", fg=typer.colors.RED)

        if mlflow_run_name and settings.observability.mlflow_enabled:
            with mlflow_run(
                settings.observability,
                run_name=mlflow_run_name,
                tags={"qa_version": qa_version, "mode": mode},
            ) as mlrun:
                log_params(
                    mlrun,
                    {
                        "k": k,
                        "mode": mode,
                        "qa_version": qa_version,
                        "check_answers": check_answers,
                    },
                )
                log_metrics(
                    mlrun,
                    {k2: v for k2, v in report.aggregate.items() if isinstance(v, int | float)},
                )
                if report.faithfulness_mean is not None:
                    log_metrics(mlrun, {"faithfulness_mean": report.faithfulness_mean})

        if out:
            out.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
            typer.echo(f"wrote {out}")
        if not report.passed:
            raise typer.Exit(code=1)
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()


@eval_app.command("calibrate")
def calibrate(
    version: Annotated[
        str | None, typer.Option(help="QA set version; default is the latest generated.")
    ] = None,
    k: Annotated[int, typer.Option()] = 10,
    min_sufficient_rate: Annotated[
        float,
        typer.Option(help="Minimum fraction of answerable queries that must stay 'sufficient'."),
    ] = 0.8,
    config_file: ConfigOpt = None,
) -> None:
    """Sweep rag_min_confidence and recommend the strictest value that doesn't start refusing real answers."""
    settings = _settings(config_file)
    engine = DuckDBEngine(settings)
    db = Database(settings)
    try:
        from akl.eval.calibration import recommend_threshold, sweep_confidence_thresholds
        from akl.eval.runner import run_eval
        from akl.lakehouse.gold import GoldStore
        from akl.lakehouse.io import LakehouseIO
        from akl.rag.service import RAGService

        io = LakehouseIO(settings, engine)
        gold = GoldStore(
            io,
            engine,
            embedding_version=settings.embedding.embedding_version,
            view_params={"chunker_version": settings.chunking.chunker_version},
        )
        qa_version = version or gold.latest_qa_version()
        if not qa_version:
            typer.secho(
                "no QA set found; run `akl-cli eval generate-qa` first", fg=typer.colors.YELLOW
            )
            raise typer.Exit(code=1)
        qa_pairs = gold.read_qa_pairs(version=qa_version).to_pylist()
        rag = RAGService(settings, engine, db)
        report = run_eval(rag, qa_pairs, principal=Principal.dev(), k=k)
        points = sweep_confidence_thresholds(
            report.per_query,
            strong_confidence=settings.retrieval.rag_strong_confidence,
            min_candidates=settings.retrieval.rag_min_candidates,
        )
        for p in points:
            typer.echo(
                f"  min_confidence={p.min_confidence:<5} sufficient_rate={p.sufficient_rate:.3f} refusal_precision={p.refusal_precision:.3f} refusal_recall={p.refusal_recall:.3f}"
            )
        recommended = recommend_threshold(points, min_sufficient_rate=min_sufficient_rate)
        typer.secho(
            f"current AKL_RAG_MIN_CONFIDENCE={settings.retrieval.rag_min_confidence}",
            fg=typer.colors.CYAN,
        )
        if recommended is not None:
            typer.secho(
                f"recommended AKL_RAG_MIN_CONFIDENCE={recommended} (keeps >= {min_sufficient_rate:.0%} of answerable queries sufficient)",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"no threshold in the sweep keeps >= {min_sufficient_rate:.0%} sufficient; corpus/QA set may need review",
                fg=typer.colors.YELLOW,
            )
    except AKLError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        engine.close()
        db.dispose()
