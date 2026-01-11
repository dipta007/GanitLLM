# Basic Makefile for GanitLLM Project

# Variables
PYTHON = python3
PIP = pip3
PROJECT_NAME = GanitLLM
VENV_NAME = .venv
PYTHON_VENV = $(VENV_NAME)/bin/python
PIP_VENV = $(VENV_NAME)/bin/pip

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup targets
.PHONY: setup
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	uv sync

.PHONY: format
format: ## Format code
	@echo "Formatting code..."
	uvx ruff format

# Cleanup targets
.PHONY: clean
clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true

.PHONY: clean-venv
clean-venv: ## Remove virtual environment
	@echo "Removing virtual environment..."
	rm -rf $(VENV_NAME)

.PHONY: clean-all
clean-all: clean clean-venv ## Clean everything including virtual environment

.PHONY: eval
eval: 
	@echo "Evaluating $(d) with model $(m)"
	PYTHONPATH=. uv run src/evaluation/eval.py --dataset $(d) --model_name $(m)

.PHONY: eval-ckpts
eval-ckpts: ## Evaluate all models on all datasets
	@echo "Evaluating all checkpoints on all datasets"
	PYTHONPATH=. uv run src/evaluation/run_ckpts.py

.PHONY: tag-difficulty
tag-difficulty: ## Tag difficulty of the dataset

.PHONY: process-dataset
process-dataset: ## Process the dataset
	@echo "filtering the dataset"
	PYTHONPATH=. uv run src/filter.py | tee logs/filter.log
	@echo "deduplicating the dataset"
	PYTHONPATH=. uv run src/deduplication.py | tee logs/deduplication.log
	@echo "minhashing the dataset"
	PYTHONPATH=. uv run src/minhash.py | tee logs/minhash.log
	@echo "decontaminating the dataset"
	PYTHONPATH=. uv run src/decontamination.py | tee logs/decontamination.log
	@echo "Tagging difficulty of the dataset"
	PYTHONPATH=. uv run src/tag_difficulty.py --dataset_name $(d)
