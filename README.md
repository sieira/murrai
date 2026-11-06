# murrAI

A conversational language model built and trained from scratch, on one
machine, to learn how LLMs work on the inside. This is a learning project
for personal knowledge, not a product; it is not meant to be state of the
art.

The model's target (doc/adr/0004): a 10-50M parameter model that matches or
beats GPT-2-small per parameter, within its own domain. Chat output will
stay crude on purpose.

## How it is built

- Layers live in `murrai/`: pure math in `domain/`, orchestration in
  `application/`, file and network access in `adapters/`, the chat REPL in
  `presentation/`, wiring in `composition/` (doc/adr/0002).
- The layer map is enforced by a tool, not by review alone (doc/adr/0003).

## Commands

```bash
make setup    # .venv + dependencies
make check    # lint + boundaries + tests; the session-end gate
make test     # full pytest suite
make test-one SCRIPT=<substring>   # one test while iterating
make lint     # ruff
make bench    # timing evidence for the training machine
```

## License

GPL-3.0 (see LICENSE).
