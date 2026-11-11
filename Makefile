.PHONY: setup check test test-one lint boundaries bench corpus chat

setup:
	uv sync

chat:
	uv run python scripts/chat.py

corpus:
	uv run python scripts/fetch_gutenberg.py

check: lint boundaries test

test:
	uv run pytest

test-one:
	uv run pytest -k "$(SCRIPT)"

lint:
	uv run ruff check murrai tests scripts
	uv run ruff format --check murrai tests scripts

boundaries:
	uv run lint-imports

bench:
	uv run python scripts/mps_sanity.py
