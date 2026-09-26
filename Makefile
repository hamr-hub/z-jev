# Z-Jev developer Makefile.
#
# Targets are intentionally simple and side-effect-free when possible.
# Each one delegates to a single ``python -m`` / shell command so it
# composes with CI and other wrappers.

SHELL := /usr/bin/env bash
ROOT := $(shell pwd)
PY ?= python3
PIP ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest -q
RUFF ?= $(PY) -m ruff

# Default goal.
.DEFAULT_GOAL := help

.PHONY: help install dev train smoke lint test clean \
        docker-build docker-up docker-down docker-logs \
        lora-tiny serve-sample

help: ## List available targets.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install z-jev (CPU).
	$(PIP) install -r requirements/cpu.txt
	$(PIP) install -e .

dev: ## Install with dev extras (pytest + ruff).
	$(PIP) install -r requirements/cpu.txt
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

lint: ## Run ruff on the package + tests.
	$(RUFF) check z_jev tests

test: ## Run the full pytest suite.
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTEST)

# ---------------------------------------------------------------------------
# Training / smoke
# ---------------------------------------------------------------------------

train: ## Train the tiny replica (mixed dataset, ~30s on CPU).
	$(PY) -m z_jev.train --out checkpoints/tiny --steps 80 --n-train 128 --n-val 32

lora-tiny: ## Run a tiny CPU LoRA training (mirrors the GLM-5 flow).
	$(PY) -m z_jev.lora_train \
	    --backbone tiny \
	    --train-file examples/train_sample.jsonl \
	    --val-file examples/train_sample.jsonl \
	    --steps 40 --batch-size 4 --lora-rank 4 --lora-alpha 8.0 \
	    --out checkpoints/lora-tiny

smoke: ## End-to-end smoke: lint + test + train + CLI infer + curl API.
	bash scripts/smoke.sh

serve-sample: ## Start the API on 127.0.0.1:8080 with the latest tiny checkpoint.
	$(PY) -m z_jev.serve --host 127.0.0.1 --port 8080 \
	    --checkpoint $$(ls -t checkpoints/tiny/model.pt 2>/dev/null | head -n1)

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the CPU Docker image.
	docker build -t z-jev:dev -f Dockerfile .

docker-up: ## Run the stack via docker compose (port 8080 by default).
	docker compose up -d --build

docker-down: ## Stop and remove the stack.
	docker compose down

docker-logs: ## Tail the compose logs.
	docker compose logs -f --tail=200 z-jev

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove caches + generated artifacts (does NOT delete checkpoints).
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
