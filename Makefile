.PHONY: setup render preview check

QUARTO_PYTHON := $(CURDIR)/.venv/bin/python

setup:
	uv sync --frozen

render:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto render

preview:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto preview

check:
	uv run ruff check assets
