.PHONY: setup notebook reset-notebooks check first-model sensor baseline format

setup:
	uv sync --locked

notebook:
	uv run jupyter lab notebooks

reset-notebooks:
	uv run python tools/reset_notebooks.py

check:
	uv lock --check
	uv run ruff check src tests tools
	uv run ruff format --check src tests tools
	uv run python -m pytest -q

first-model:
	uv run python -m ml_sau.first_model

sensor:
	uv run python -m ml_sau.sensor

baseline:
	uv run python -m ml_sau.baseline \
		--config configs/baseline.json \
		--output reports/runs/s3-baseline

format:
	uv run ruff check --fix src tests tools
	uv run ruff format src tests tools
