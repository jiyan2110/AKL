#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_USER:?}"
: "${AKL_DB_NAME:?}" "${AKL_DB_API_USER:?}" "${AKL_DB_API_PASSWORD:?}"
: "${AKL_DB_PIPELINE_USER:?}" "${AKL_DB_PIPELINE_PASSWORD:?}"
: "${AIRFLOW_DB_NAME:?}" "${AIRFLOW_DB_USER:?}" "${AIRFLOW_DB_PASSWORD:?}"
: "${MLFLOW_DB_NAME:?}" "${MLFLOW_DB_USER:?}" "${MLFLOW_DB_PASSWORD:?}"

psql_admin() {
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres "$@"
}

create_role() {
  psql_admin -v role="$1" -v pw="$2" <<-'SQL'
	SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'role', :'pw')
	WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'role')
	\gexec
	SQL
}

create_db() {
  psql_admin -v db="$1" -v owner="$2" <<-'SQL'
	SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'owner')
	WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db')
	\gexec
	SQL
}

echo "[akl-pg-init] creating roles"
create_role "$AKL_DB_PIPELINE_USER" "$AKL_DB_PIPELINE_PASSWORD"
create_role "$AKL_DB_API_USER" "$AKL_DB_API_PASSWORD"
create_role "$AIRFLOW_DB_USER" "$AIRFLOW_DB_PASSWORD"
create_role "$MLFLOW_DB_USER" "$MLFLOW_DB_PASSWORD"

echo "[akl-pg-init] creating databases"
create_db "$AKL_DB_NAME" "$AKL_DB_PIPELINE_USER"
create_db "$AIRFLOW_DB_NAME" "$AIRFLOW_DB_USER"
create_db "$MLFLOW_DB_NAME" "$MLFLOW_DB_USER"

echo "[akl-pg-init] granting ${AKL_DB_API_USER} access on ${AKL_DB_NAME}"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$AKL_DB_NAME" <<-SQL
	GRANT CONNECT ON DATABASE "${AKL_DB_NAME}" TO "${AKL_DB_API_USER}";
	GRANT USAGE, CREATE ON SCHEMA public TO "${AKL_DB_PIPELINE_USER}";
	GRANT USAGE ON SCHEMA public TO "${AKL_DB_API_USER}";
	ALTER DEFAULT PRIVILEGES FOR ROLE "${AKL_DB_PIPELINE_USER}" IN SCHEMA public
	  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "${AKL_DB_API_USER}";
	ALTER DEFAULT PRIVILEGES FOR ROLE "${AKL_DB_PIPELINE_USER}" IN SCHEMA public
	  GRANT USAGE, SELECT ON SEQUENCES TO "${AKL_DB_API_USER}";
	SQL

echo "[akl-pg-init] done"
