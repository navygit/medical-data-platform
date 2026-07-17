# Developer entrypoints. `make help` lists everything.
#
# The contract: `make demo` takes a clean clone to a full set of artifacts with
# no credentialed downloads and no GPU. If that ever breaks, the repo is broken.

.PHONY: help install install-ml synth brats mimic lits demo test test-fast lint format typecheck clean ci

PYTHON ?= python
VENV   ?= .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package plus dev tooling (core deps only)
	$(PYTHON) -m pip install -e ".[dev]"

install-ml:  ## Add the deep-learning extra (torch, MONAI); ~2 GB
	$(PYTHON) -m pip install -e ".[ml]"

synth:  ## Generate the synthetic corpus for all three datasets
	$(PYTHON) scripts/generate_synthetic_data.py --dataset all --n 12

brats:  ## Run the BraTS pipeline
	$(PYTHON) -m pipelines.brats.run --config configs/brats.yaml

mimic:  ## Run the MIMIC-CXR multimodal pipeline
	$(PYTHON) -m pipelines.mimic.run --config configs/mimic.yaml

lits:  ## Run the LiTS governance pipeline
	$(PYTHON) -m pipelines.lits.run --config configs/lits.yaml

demo: synth brats mimic lits  ## Full demo: synth data -> all three pipelines
	@echo ""
	@echo "Done. Artifacts:"
	@echo "  outputs/brats/QC_REPORT.html    QC report"
	@echo "  outputs/brats/figures/          slice overlays"
	@echo "  outputs/lits/dataset_card.md    generated dataset card"
	@echo "  outputs/mimic/PHI_AUDIT.csv     PHI audit"
	@echo "  releases/*/v1.0.0/manifest.json versioned releases"

test:  ## Run the full test suite with coverage
	$(PYTHON) -m pytest tests/ -v --cov=common --cov=pipelines --cov-report=term-missing

test-fast:  ## Run unit tests only (skip end-to-end)
	$(PYTHON) -m pytest tests/ -q -m "not slow"

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check common pipelines scripts tests

format:  ## Format with black and autofix with ruff
	$(PYTHON) -m black common pipelines scripts tests
	$(PYTHON) -m ruff check --fix common pipelines scripts tests

typecheck:  ## Type-check with mypy
	$(PYTHON) -m mypy common pipelines

ci: lint typecheck test  ## Everything CI runs

clean:  ## Remove generated data, artifacts and caches
	rm -rf data outputs releases mlruns .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
