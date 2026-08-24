.PHONY: setup lock-check lint test-python render preview check verify

QUARTO_PYTHON := $(CURDIR)/.venv/bin/python

setup:
	uv sync --locked
	uv sync --locked --project starter

lock-check:
	uv lock --check
	uv lock --check --project starter

lint:
	uv run ruff check assets tests starter/src starter/tests
	uv run ruff format --check assets tests starter/src starter/tests

test-python:
	uv run python -m pytest -q tests
	uv run --locked --project starter python -m pytest -q starter/tests

render:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto render

preview:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto preview

check: lock-check lint test-python

verify: check render
