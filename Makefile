.PHONY: setup render preview check sites-dist

QUARTO_PYTHON := $(CURDIR)/.venv/bin/python

setup:
	uv sync --frozen

render:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto render

preview:
	QUARTO_PYTHON="$(QUARTO_PYTHON)" quarto preview

check:
	uv run ruff check assets

sites-dist: render
	./scripts/build-sites-static.sh
