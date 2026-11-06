# ADR-0002: Pragmatic DDD and hexagonal architecture for murrAI

- Status: Accepted
- Date: 2026-11-06

## Context

murrAI is a learning project: the author reads and understands everything
built here. The valuable core is pure math (model architecture, sampling,
tokenizer). Training needs dataset downloads and checkpoint files; chatting
needs a REPL. If I/O and interface code mix into the math, unit tests need
network and GPU, and the learning core drowns in plumbing.

## Decision

We adopt a pragmatic subset of DDD inside a hexagonal (ports and adapters)
architecture:

1. **Hexagon.** Five code layers in `murrai/`:
   - `domain/` — pure math: tokenizer, model, sampling. Torch tensor math is
     allowed; torch I/O, network access, and filesystem writes are not.
   - `application/` — orchestration (training runs, generation sessions);
     depends on domain plus ports.
   - `adapters/` — port implementations: dataset download, checkpoint
     files, and fakes for tests.
   - `presentation/` — renders state and samples; it computes no math.
     The chat (REPL, stage picker, chat session) is a separable unit
     inside it, `presentation/chat/`, touching the core only through the
     `LanguageModel` port, so it could later be extracted into its own
     project.
   - `composition/` — wires concrete adapters into ports.
2. **Ports.** `DataSource` (training corpora), `LanguageModel` (the chat's
   view of any model), and `CheckpointStore` (model persistence) start the
   port set; new I/O needs enter through a port.
3. **DDD subset we use:** a shared vocabulary (doc/ddd/glossary.md) binding
   identifiers across code, tests, and docs; domain invariants where they
   exist (a tokenizer's encode/decode round trip is one).
4. **DDD we skip:** repository ceremony for single-implementation ports,
   CQRS.

### Acceptance tests for this decision

- Domain tests pass with the network disabled and no GPU present.
- A fake `DataSource` and a fake `CheckpointStore` drive a tiny training
  run end to end; if that stops working, the hexagon is nominal, not real.
- The REPL renders and samples only; no math in presentation.

## Enforcement

Boundaries are machine-enforced, not aspirational:
[ADR-0003](0003-machine-enforced-boundaries.md) makes `make boundaries` fail
on any dependency arrow outside [doc/ddd/layers.md](../ddd/layers.md).

## Consequences

- The learning core stays readable and testable anywhere, offline.
- Slightly more files and indirection than a single-script layout.
- Framework or infrastructure changes touch adapters and composition only.
