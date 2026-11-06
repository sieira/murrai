# ADR-0003: Machine-enforced boundaries

- Status: Accepted
- Date: 2026-11-06

## Context

ADR-0002 defines a layer map. Boundaries that only reviewers enforce rot:
one urgent PR crosses a layer, the next copies it. The sibling project chloe
hand-rolled a checker because GDScript had no boundary tooling. Python has
import-linter, a maintained tool for exactly this contract.

## Decision

- import-linter enforces the dependency contract in
  [doc/ddd/layers.md](../ddd/layers.md).
- Configuration lives at the repo root; `make boundaries` runs it; `make
  check` includes it.
- A violation is fixed by moving the import, or by a documented exception in
  the config. Bypassing or deleting checks is not a fix (AGENTS.md rule 10).

## Consequences

- One config file to maintain as layers evolve.
- Layer changes are ADR material: they edit the contract and its enforcement
  together.
