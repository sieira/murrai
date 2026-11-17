.PHONY: setup check test test-one lint boundaries bench bench-train train corpus corpus-theme corpus-tinystories chat

setup:
	uv sync

chat:
	uv run python scripts/chat.py

corpus:
	uv run python scripts/fetch_gutenberg.py

corpus-tinystories:
	uv run python scripts/fetch_tinystories.py

corpus-theme:
	uv run python scripts/fetch_gutenberg.py $(THEME)

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

bench-train:
	uv run python scripts/train_bench.py

train:
	uv run python scripts/train.py $(STEPS)
