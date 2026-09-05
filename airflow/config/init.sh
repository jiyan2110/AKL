#!/usr/bin/env bash
# One-shot Airflow bootstrap: migrate metadata DB, create admin user, import pools (idempotent).
set -euo pipefail
airflow db migrate
airflow users create \
  --username "${AIRFLOW_ADMIN_USER:-admin}" --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
  --firstname AKL --lastname Admin --role Admin --email admin@example.com >/dev/null 2>&1 || true
airflow pools import /opt/airflow/config/pools.json
echo "airflow init complete"
