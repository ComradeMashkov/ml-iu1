.PHONY: setup check format

setup:
	uv sync --locked

check:
	uv lock --check
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run python -m pytest -q

first-model:
	uv run python -m ml_sau.first_model

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests
