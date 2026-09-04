#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_USER:?}" "${AKL_DB_NAME:?}"

echo "[akl-pg-init] installing extensions in ${AKL_DB_NAME}"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$AKL_DB_NAME" <<-'SQL'
	CREATE EXTENSION IF NOT EXISTS "pgcrypto";
	CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
	SQL

echo "[akl-pg-init] extensions done"
