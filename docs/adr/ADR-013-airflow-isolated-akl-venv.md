# ADR-013 — Airflow tasks run the `akl` package in an isolated virtualenv

**Status:** Accepted (Batch F, Milestones 37–42)

## Context
Apache Airflow 2.10 pins `sqlalchemy<2.0`; the `akl` package requires SQLAlchemy 2.x (typed ORM,
psycopg3). The two cannot be installed into one interpreter without downgrading the project's
data layer. The PRD requires DAG files to be thin and all logic to live in `akl.pipelines`.

## Decision
- The Airflow image (`docker/airflow/Dockerfile`) builds a second interpreter at
  `/opt/akl-venv` with `pip install -e /opt/akl` (no Airflow inside it).
- Every DAG task is `@task.external_python(python=AKL_PYTHON, expect_airflow=False)` and calls
  one JSON-in/JSON-out function in `akl.pipelines.airflow_tasks`. Task bodies contain only that
  import and call; closure variables are never used (the operator ships function *source*, not
  the enclosing scope).
- `pipeline_runs`/`task_runs` bookkeeping, quality gates (`AKL-E7001`) and Dataset publishing
  live in `akl`, so `akl-cli pipeline …` (`make pipeline`) executes the identical stages without
  Airflow — used for local runs, tests and CI.
- DAG-integrity tests run wherever Airflow is importable (scheduler container via
  `make dags-test`, CI) and skip in the host venv (Airflow does not run on Windows).
- Outputs of mapped tasks are materialised by an Airflow-side `@task` before being passed to an `external_python` task; only JSON types cross the boundary.
## Consequences
- + No dependency compromise on either side; Airflow can be upgraded independently.
- + The same entrypoints power CLI, API background jobs and DAGs (one code path to test).
- − XCom payloads must stay JSON-serialisable (enforced by returning plain dicts).
- − A second interpreter adds ~400 MB to the image; the `akl` source is bind-mounted read-only
  in dev so code changes do not require a rebuild (dependency changes do: `make airflow-build`).
