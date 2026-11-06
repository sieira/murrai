# ADR-0001: Record architecture decisions

- Status: Accepted
- Date: 2026-11-06

## Context

This repository is built agent-assisted and serves a single human learning
how LLMs work internally. Architectural reasoning that lives only in chat
transcripts or one agent's memory is lost, and future changes risk silently
violating past decisions.

## Decision

We record architecture and tooling decisions as lightweight Architecture
Decision Records (ADRs) in `doc/adr/`.

- One numbered file per decision: `NNNN-kebab-case-title.md`.
- Format: Status / Date / Context / Decision / Consequences (Nygard-style).
- ADRs are living records, not immutable ones. A decision that evolves is
  recorded by editing its ADR; a new file appears only for a genuinely new
  decision. The record does not grow in number or size by default.
- An ADR edit enforces the decision in the artifact in the same increment:
  the change that edits the ADR also edits the code, config, or docs that
  implement it. Record and enforced reality never drift apart.
- Any change that alters the project structure, the layer map, the tooling,
  or the working rules requires an ADR edit or author approval. Small
  implementation choices do not.

## Consequences

- "Why is it like this?" is answered by reading `doc/adr/`, not by guessing.
- A small process tax on structural changes; none on ordinary coding.
- AGENTS.md and code review reference the relevant ADR instead of restating
  rationale.
