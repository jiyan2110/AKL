# ============================================================================
# AKL — Enterprise AI Knowledge Lakehouse
# Developer entry point (PRD §11.6). Requires: uv, docker, docker compose,
# GNU Make, and on Windows: Git for Windows (provides bash, grep, awk, find).
# ============================================================================

# --- Portable shell selection --------------------------------------------
# Linux/macOS: /bin/bash. Windows: Git Bash. Override with: make SHELL=...
ifeq ($(OS),Windows_NT)
	GIT_BASH := $(firstword $(wildcard C:/Program\ Files/Git/bin/bash.exe C:/Program\ Files\ (x86)/Git/bin/bash.exe $(LOCALAPPDATA)/Programs/Git/bin/bash.exe))
	ifeq ($(GIT_BASH),)
		$(error Git Bash not found. Install Git for Windows or run: make SHELL="C:/path/to/bash.exe")
	endif
	SHELL := $(GIT_BASH)
else
	SHELL := /bin/bash
endif
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help

UV            ?= uv
PY            := $(UV) run python
COMPOSE_BASE  := docker compose -f docker-compose.yml
COMPOSE_DEV   := $(COMPOSE_BASE) -f docker-compose.dev.yml
COMPOSE_PROD  := $(COMPOSE_BASE) -f docker-compose.prod.yml
YAMLLINT_CFG  := {extends: default, rules: {line-length: {max: 140}, document-start: disable, truthy: disable}}

# Milestone that implements a not-yet-available target (edited as we go).
define not_yet
	@echo "[$(1)] not yet implemented — arrives in Milestone $(2)."
endef

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
.PHONY: install
install: ## Create .venv and install project + dev dependencies
	$(UV) sync --extra dev

.PHONY: hooks
hooks: ## Install pre-commit hooks
	$(UV) run pre-commit install

.PHONY: lint
lint: ## Ruff lint + format check + mypy + yamllint
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy
	$(UV) run yamllint -d '$(YAMLLINT_CFG)' .

.PHONY: fmt
fmt: ## Auto-fix lint issues and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: test
test: ## Run unit tests with coverage (exit code 5 = no tests collected, allowed)
	$(UV) run pytest -m "not component and not integration and not api and not eval and not slow" --cov --cov-report=term-missing || [ $$? -eq 5 ]

.PHONY: test-unit
test-unit: ## Unit tests only
	$(UV) run pytest -m unit

.PHONY: test-component
test-component: ## Component tests (require running services)
	$(UV) run pytest -m component

.PHONY: test-integration
test-integration: ## Integration tests against the Compose stack
	$(UV) run pytest -m integration

.PHONY: test-api
test-api: ## API contract tests
	$(UV) run pytest -m api

# ----------------------------------------------------------------------------
# Docker Compose stack
# ----------------------------------------------------------------------------
.PHONY: env
env: ## Create .env from .env.example if missing
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example — review the passwords."; else echo ".env exists"; fi

.PHONY: up
up: env ## Start the dev stack (detached)
	$(COMPOSE_DEV) up -d --remove-orphans

.PHONY: down
down: ## Stop the stack, keep volumes
	$(COMPOSE_DEV) down --remove-orphans

.PHONY: nuke
nuke: ## Stop the stack and delete volumes (destroys all data)
	@read -r -p "This deletes ALL local data volumes. Type 'yes' to continue: " ans; [ "$$ans" = "yes" ]
	$(COMPOSE_DEV) down --volumes --remove-orphans

.PHONY: logs
logs: ## Tail stack logs (SERVICE=name to filter)
	$(COMPOSE_DEV) logs -f --tail=200 $(SERVICE)

.PHONY: ps
ps: ## Show container status and health
	$(COMPOSE_DEV) ps

WAIT_TIMEOUT ?= 180
.PHONY: wait
wait: ## Wait until postgres, minio, qdrant are healthy and minio-init succeeded
	@echo "Waiting up to $(WAIT_TIMEOUT)s for services..."
	@deadline=$$(( $$(date +%s) + $(WAIT_TIMEOUT) )); \
	for c in akl-postgres akl-minio akl-qdrant; do \
	  until [ "$$(docker inspect -f '{{.State.Health.Status}}' $$c 2>/dev/null)" = "healthy" ]; do \
	    if [ $$(date +%s) -ge $$deadline ]; then echo "TIMEOUT waiting for $$c"; docker inspect -f '{{json .State.Health}}' $$c; exit 1; fi; \
	    printf '.'; sleep 3; \
	  done; echo " $$c healthy"; \
	done; \
	until [ "$$(docker inspect -f '{{.State.Status}}' akl-minio-init 2>/dev/null)" = "exited" ]; do \
	  if [ $$(date +%s) -ge $$deadline ]; then echo "TIMEOUT waiting for akl-minio-init"; exit 1; fi; \
	  printf '.'; sleep 3; \
	done; \
	code=$$(docker inspect -f '{{.State.ExitCode}}' akl-minio-init); \
	if [ "$$code" != "0" ]; then echo "akl-minio-init FAILED (exit $$code)"; docker logs akl-minio-init; exit 1; fi; \
	echo " akl-minio-init ok"; echo "All services ready."

# ----------------------------------------------------------------------------
# Data & pipelines
# ----------------------------------------------------------------------------
.PHONY: seed
seed: ## Ingest → chunk → Gold → embed → Qdrant sync + BM25 (full offline pipeline)
	$(UV) run akl-cli ingest run
	$(UV) run akl-cli chunk run
	$(UV) run akl-cli embed run
	$(UV) run akl-cli qdrant sync
	$(UV) run akl-cli embed status
	$(UV) run akl-cli bm25 status

.PHONY: pipeline
pipeline: ## Run the five pipeline stages sequentially via the task entrypoints the DAGs use (no Airflow needed)
	$(UV) run akl-cli pipeline run-all

.PHONY: airflow-build airflow-up airflow-down airflow-logs dags-test dags-unpause
airflow-build: ## Build the Airflow image (Airflow 2.10 + isolated akl venv)
	$(COMPOSE_DEV) build airflow-init

airflow-up: ## Start Airflow (init → scheduler/webserver/triggerer); UI http://localhost:8080 (admin/admin)
	$(COMPOSE_DEV) up -d airflow-init airflow-scheduler airflow-webserver airflow-triggerer

airflow-down: ## Stop Airflow services (data services stay up)
	$(COMPOSE_DEV) stop airflow-scheduler airflow-webserver airflow-triggerer

airflow-logs: ## Tail scheduler logs
	$(COMPOSE_DEV) logs -f --tail=200 airflow-scheduler

dags-test: ## Import all DAGs inside the scheduler container, then run akl_ingestion once end-to-end
	$(COMPOSE_DEV) exec airflow-scheduler python -c "from airflow.models import DagBag; b=DagBag('/opt/airflow/dags', include_examples=False); assert not b.import_errors, b.import_errors; print('DAGs OK:', sorted(b.dags))"
	$(COMPOSE_DEV) exec airflow-scheduler airflow dags test akl_ingestion

dags-unpause: ## Unpause all AKL DAGs
	$(COMPOSE_DEV) exec airflow-scheduler bash -c 'for d in akl_ingestion akl_chunking akl_embedding akl_qdrant_sync akl_maintenance; do airflow dags unpause $$d; done'

.PHONY: token
token: ## Mint a development JWT (needs AKL_JWT_SECRET)
	$(UV) run akl-cli auth mint-token --user dev --groups eng --levels public,internal,restricted --roles admin

api: ## Run the FastAPI gateway locally (http://localhost:8000/docs)
	$(UV) run akl-cli api serve --reload

.PHONY: query
query: ## Hybrid search: make query Q="..."
	$(UV) run akl-cli search "$(Q)"

ask: ## Cited answer (extractive until the LLM lands): make ask Q="..."
	$(UV) run akl-cli ask "$(Q)"

.PHONY: bench
bench: ## Run benchmark harness
	$(call not_yet,bench,55)

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
