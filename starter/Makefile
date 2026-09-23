.PHONY: setup notebook reset-notebooks check first-model sensor s01 s02 s03 s04 format

setup:
	uv sync --locked

notebook:
	uv run jupyter lab notebooks

reset-notebooks:
	uv run python tools/reset_notebooks.py

check:
	uv lock --check
	uv run ruff check src tests tools lessons
	uv run ruff format --check src tests tools lessons
	uv run python -m pytest -q

first-model:
	uv run python lessons/s01.py

sensor:
	uv run python lessons/s02.py

s01:
	uv run python lessons/s01.py

s02:
	uv run python lessons/s02.py

s03:
	uv run python lessons/s03.py

s04:
	uv run python lessons/s04.py

format:
	uv run ruff check --fix src tests tools lessons
	uv run ruff format src tests tools lessons
