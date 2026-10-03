.PHONY: help install dev test lint demo run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (editable)
	python -m pip install -e .

dev:  ## Install with dev + hub extras
	python -m pip install -e ".[dev,hub]"

test:  ## Run the test suite
	python -m pytest

lint:  ## Lint with ruff
	ruff check src tests

demo:  ## End-to-end run with the offline mock provider (no API key needed)
	sft-forge run -c configs/demo_mock.yaml --set run.name=demo

run:  ## Real run against your configured API endpoint
	sft-forge run -c configs/general.yaml

clean:  ## Remove generated artifacts
	rm -rf runs .cache dist build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
