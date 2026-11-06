# SPEC.md — murrAI

Build and train a conversational LLM from scratch, on one machine, to learn the
internals. The project root is this repository. The model's name is murrAI.

Project facts:
- Root: this folder.
- Machine: Apple M4, 32 GB unified memory, Metal GPU. Training runs on PyTorch MPS.
- License: GPL-3.0. The README states it is a learning project for personal knowledge.
- Hosting: public GitHub repo, name `murrai` under the author's account unless the
  author says otherwise at creation time. Squash-merge PRs, Conventional Commits.

## Architecture (the bases)

Hexagonal, after the sibling project chloe /Users/luis.sieira/sandbox/chloe:

- `murrai/domain/` pure math: model, sampling, tokenizer. No I/O, no network.
- `murrai/application/` training orchestration; depends on domain and ports.
- `murrai/adapters/` dataset download, checkpoint files.
- `murrai/presentation/` chat REPL.
- `murrai/composition/` wiring.
- `tests/` pytest suite.

Machine-enforced boundaries via import-linter (`make boundaries`); the map
lives in `doc/ddd/layers.md`. `doc/ddd/glossary.md` holds the ubiquitous
language and doubles as study notes; both bind.

Docs: `doc/adr/` (Nygard format, living records edited together with the
artifacts they govern), README, AGENTS.md.

## Binding rules (AGENTS.md carries the full set)

- TDD: a failing test is shown red before green for every deterministic change
  (tokenizer round-trips, attention shapes/masking, seeded sampling, checkpoint
  I/O, REPL formatting). Training runs verify by artifacts: loss curves and
  sample outputs. `make test-one` while iterating; `make check` once, at
  session end.
- Small increments: one session = one branch/worktree; PR-only merges; large
  diffs split first.
- Comments: none by default; only a non-obvious why earns a line.
- Deletion first: no receipts of removed things, no records of what is not,
  unless it adds something. Deleting beats editing; editing beats adding.
- Definition of done: red shown before green, `make check` green, claims carry
  evidence (test output, loss curve, sample text), plain language everywhere.
- Author interaction: agents never implement what the author has not clearly
  understood. Doubt stops work and becomes a direct question in plain
  language, with context, through the opencode question tool, one question at
  a time; the run waits for the answer. Scope expansion is a proposal back,
  never an action.
- Unslop, systematic: installed from
  https://github.com/cursor/plugins/blob/main/pstack/skills/unslop/SKILL.md
  (raw fetch into ~/.config/opencode/skills/unslop/). Every author-facing
  artifact (questions, reports, README, ADRs, glossary, PR bodies) gets an
  unslop pass. Prose only; code cleanliness rests on the rules above.

## Roster (process, kept out of ADRs on purpose)

- pm: phases, backlog, ADR drafting, session plans.
- builder: dev+tester fused. Owns `murrai/` code and `tests/`. Inherits dev's
  discipline (TDD cadence, scope discipline, stop rule, labeled hypotheses)
  and tester's ethics as hard rules: never weaken a test to pass, sabotage-
  check every green ("what fix would make this fail?"), coverage cuts are
  author decisions. If a tautological test or pruned coverage ships, split
  the agent that day.
Both live in `.opencode/agent/`. Process lives here and in agent files, not
in doc/adr.

## ADRs to write in Phase 0

- 0001 record architecture decisions (meta-ADR)
- 0002 pragmatic DDD + hexagonal architecture for murrAI
- 0003 machine-enforced boundaries (import-linter)
- 0004 target: match-or-beat GPT-2-small per parameter (details below)

## Target (ADR-0004, pass-or-fail)

A 10-50M parameter murrAI, trained on this machine, with at-or-better
perplexity per parameter than GPT-2-small on a named held-out corpus, and
grammatical fluency matching it on its home domain. Scope limit, recorded:
within-domain coherence only; world knowledge and general chat are out of
scope. Chat output is expected to remain crude; that is by design.

## Phases (each = small PRs; ADRs only at structural decisions)

0. Bases: git init, GitHub repo, .gitignore (opencode.json, .venv/, .DS_Store),
   LICENSE, README, AGENTS.md, ADRs 0001-0004, doc/ddd scaffolding, uv project
   (Python 3.12, torch, pytest, ruff, import-linter), Makefile (setup, check,
   test, test-one, lint, boundaries, train, chat), .opencode/agent/pm.md +
   builder.md, unslop install, MPS sanity benchmark.
1. Foundations: bigram LM; micrograd (own autograd engine, pure Python).
2. MLP LM: embeddings, context windows, training loop, LR schedules.
3. Transformer: attention -> blocks -> full GPT, char-level. First murrAI.
4. Tokenizer: own BPE (minbpe-style); ADR for the swap.
5. Scale: 10-50M params on TinyStories, overnight MPS runs; ADR-0004 evidence;
   weights-licensing ADR if a checkpoint is published.
6. Conversation: train from random init on chat-formatted data; the REPL.
7. Polish: KV-cache, optional Gradio UI, writeup.

Reference material (peek only, never copy wholesale): Karpathy's
Neural Networks: Zero to Hero, nanoGPT, minbpe.

## First session start

1. Write AGENTS.md from this spec.
2. git init; .gitignore first commit.
3. Create public repo `murrai`; push.
4. Scaffold uv project + Makefile + ruff + import-linter config.
5. Write ADRs 0001-0004; write doc/ddd/layers.md + glossary.md seeds.
6. Write .opencode/agent/pm.md and builder.md.
7. Install unslop skill.
8. MPS sanity benchmark: torch matmul on Metal, timing recorded as evidence.

Starting assumptions for phase 1 work are recorded per phase in doc/adr at
decision time, not in advance.
