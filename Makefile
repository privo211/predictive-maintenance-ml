# ===========================================================================
# Predictive Maintenance Platform — Build & Operations
# ===========================================================================
# Usage:
#   make setup          Bootstrap the project (install deps, pre-commit hooks)
#   make generate-data  Generate synthetic training data
#   make train          Train the XGBoost failure prediction model
#   make serve          Start the FastAPI inference server
#   make test           Run the full test suite
#   make lint           Run all linters (ruff + mypy)
#   make docker-up      Start all services via Docker Compose
#   make docker-down    Stop and remove all Docker services
#   make clean          Remove build artifacts, caches, and temp files
# ===========================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Python settings
PYTHON := python3.12
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest

# Paths
SRC_DIR := src
TEST_DIR := tests
CONFIG_DIR := config
NOTEBOOK_DIR := notebooks
LOG_DIR := logs
MODEL_DIR := models
DATA_DIR := data

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: setup generate-data train serve test lint format typecheck \
        docker-up docker-down clean help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup:  ## Install dependencies and pre-commit hooks
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	pre-commit install --install-hooks
	@mkdir -p $(LOG_DIR) $(MODEL_DIR) $(DATA_DIR)
	@echo "[setup] Project bootstrapped successfully."

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------
generate-data:  ## Generate synthetic sensor data for training
	$(PYTHON) -m scripts.generate_data \
		--output-dir $(DATA_DIR) \
		--equipment-count 50 \
		--days 90 \
		--failure-rate 0.02 \
		--seed 42

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
train:  ## Train the XGBoost failure prediction model
	$(PYTHON) -m scripts.train_model \
		--data-dir $(DATA_DIR) \
		--model-dir $(MODEL_DIR) \
		--config $(CONFIG_DIR)/model_config.yaml

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------
serve:  ## Start the FastAPI inference server
	$(PYTHON) -m scripts.serve_api \
		--host $(PMP_API_HOST) \
		--port $(PMP_API_PORT) \
		--workers $(PMP_API_WORKERS)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:  ## Run the full test suite with coverage
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

test-unit:  ## Run unit tests only
	$(PYTEST) -m unit

test-integration:  ## Run integration tests only
	$(PYTEST) -m integration

test-validation:  ## Run model / data validation tests only
	$(PYTEST) -m validation

# ---------------------------------------------------------------------------
# Linting & Type Checking
# ---------------------------------------------------------------------------
lint: format-check lint-check typecheck  ## Run all linters

format:  ## Auto-format code with ruff
	$(RUFF) format $(SRC_DIR) $(TEST_DIR) scripts/

format-check:  ## Check formatting without applying changes
	$(RUFF) format --check $(SRC_DIR) $(TEST_DIR) scripts/

lint-check:  ## Run ruff linter
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) scripts/

lint-fix:  ## Auto-fix linting issues
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR) scripts/

typecheck:  ## Run mypy static type checker
	$(MYPY) $(SRC_DIR)

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-up:  ## Start all services via Docker Compose
	docker compose up -d --build
	@echo "[docker] Services started. API: http://localhost:8000"

docker-down:  ## Stop and remove Docker services
	docker compose down --volumes --remove-orphans

docker-logs:  ## Tail Docker Compose logs
	docker compose logs -f

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:  ## Remove build artifacts, caches, and temp files
	@echo "[clean] Removing build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf $(SRC_DIR)/**/__pycache__ $(TEST_DIR)/**/__pycache__
	rm -rf $(NOTEBOOK_DIR)/.ipynb_checkpoints
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf $(LOG_DIR)/*.log*
	rm -rf $(MODEL_DIR)/*
	rm -rf $(DATA_DIR)/*
	@echo "[clean] Done."

clean-all: clean  ## Clean everything including Docker volumes
	docker compose down --volumes --remove-orphans --rmi all 2>/dev/null || true
	rm -rf .venv/
	@echo "[clean-all] Done."
