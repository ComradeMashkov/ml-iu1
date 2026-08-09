.PHONY: setup check format

setup:
	uv sync --locked

check:
	uv lock --check
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run pytest -q

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests
