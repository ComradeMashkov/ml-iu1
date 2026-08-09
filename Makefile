.PHONY: setup lock-check lint test-python test-js render preview check verify

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
	uv run pytest -q tests
	uv run --locked --project starter pytest -q starter/tests

test-js:
	node --test tests/flight-mission-core.test.mjs

render:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto render

preview:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto preview

check: lock-check lint test-python test-js

verify: check render
