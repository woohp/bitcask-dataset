.PHONY: format fformat lint test

python_files = src tests

format:
	uv run ruff check --fix-only --ignore F401 $(python_files)
	uv run ruff format $(python_files)

fformat:
	uv run ruff check --fix-only $(python_files)
	uv run ruff format $(python_files)

lint:
	uv run ruff check $(python_files)
	uv run mypy $(python_files)

test:
	uv run pytest
