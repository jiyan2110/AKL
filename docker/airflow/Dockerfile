# AKL Airflow image (PRD §7.1, ADR-013): Airflow 2.10 + an isolated `akl` virtualenv.
#
# Airflow 2.x pins SQLAlchemy 1.4; the akl package needs SQLAlchemy 2. The two never share an
# interpreter: DAG tasks run through @task.external_python(python=/opt/akl-venv/bin/python).
FROM apache/airflow:2.10.5-python3.12

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client git build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- akl venv ------------------------------------------------------------------------------
COPY --chown=airflow:root pyproject.toml README.md /opt/akl/
COPY --chown=airflow:root akl /opt/akl/akl

RUN python -m venv /opt/akl-venv \
 && /opt/akl-venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/akl-venv/bin/pip install --no-cache-dir -e /opt/akl \
 && chown -R airflow:root /opt/akl-venv

USER airflow

# --- airflow side ------------------------------------------------------------------------------
# Only what the DAG files import at parse time (PyYAML is already part of the Airflow image).
ENV AKL_PYTHON=/opt/akl-venv/bin/python \
    AKL_DAG_CONFIG_DIR=/opt/airflow/configs/dags \
    AKL_CONFIG_DIR=/opt/akl/configs \
    PYTHONPATH=/opt/airflow/plugins
