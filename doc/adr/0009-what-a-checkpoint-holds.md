# ADR-0009: What a checkpoint holds

- Status: Accepted
- Date: 2026-11-13

## Context

A checkpoint holds a model kind, the tokenizer's merge list, and the
weights (ADR-0008). We want a checkpoint to tell its run's story: when
it was trained, on which corpus, with which settings, and how well the
model did. The chat menu shows a load command for every checkpoint,
and we want each command to carry those facts instead of one generic
line.

## Decision

A checkpoint holds four things: the model kind, the tokenizer's merge
list, the weights, and a record of its run. The record is a frozen
dataclass, `CheckpointRecord`. The store keeps it as a small JSON file
beside the weights file, so reading the record never opens the
weights. Its fields, each written when the run produced them:
`saved_at` (ISO time, UTC), `corpus` (the corpus directory as given),
`seed`, `steps`, `batch`, `learning_rate`, `final_loss` (the last loss
in the curve), and `held_out_perplexity`. The counted bigram has no
steps and no loss curve; the neural bigram has no held-out
perplexity. A field the run did not produce reads as none.

The record is assembled in composition, where the settings and the run
result live. A trainer hands its keeper the finished model, as before;
the builder wraps that keeper to add the settings it knows, and the
session writes the record again once the result exists, so the run
numbers join it. Both writes happen inside the training process,
before anything crosses the pipe to the chat (ADR-0005).

## Consequences

- The weights file is written once. The record file is written twice
  at the end of a run: once with the settings, once when the run
  numbers are known.
- A run whose metrics fail keeps its checkpoint, and its record keeps
  the settings while missing the numbers the failed metrics would
  have supplied.
