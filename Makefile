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
.PHONY: up
up: ## Start the dev stack
	$(call not_yet,up,2)

.PHONY: down
down: ## Stop the stack, keep volumes
	$(call not_yet,down,2)

.PHONY: nuke
nuke: ## Stop the stack and delete volumes (destroys all data)
	$(call not_yet,nuke,2)

.PHONY: logs
logs: ## Tail stack logs
	$(call not_yet,logs,2)

.PHONY: wait
wait: ## Wait until all services are healthy
	$(call not_yet,wait,2)

.PHONY: ps
ps: ## Show container status
	$(call not_yet,ps,2)

# ----------------------------------------------------------------------------
# Data & pipelines
# ----------------------------------------------------------------------------
.PHONY: seed
seed: ## Upload example corpus and run pipelines
	$(call not_yet,seed,32)

.PHONY: pipeline
pipeline: ## Run the five DAGs sequentially via CLI
	$(call not_yet,pipeline,42)

.PHONY: token
token: ## Mint a development JWT
	$(call not_yet,token,31)

.PHONY: query
query: ## Ask a question: make query Q="..."
	$(call not_yet,query,34)

.PHONY: bench
bench: ## Run benchmark harness
	$(call not_yet,bench,55)

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
