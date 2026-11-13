# ADR-0008: What a checkpoint holds

- Status: Accepted
- Date: 2026-11-13

## Context

A checkpoint holds a model kind, the character list, and the weights.
A checkpoint opened weeks later cannot say when it was trained, on which
corpus, with which settings, or how well the model did. The chat menu
shows every load command with the same generic line. The training bench
script already keeps its run notes in a separate JSON file next to the
checkpoint they belong to.

## Decision

A checkpoint holds four things: the model kind, the character list, the
weights, and an optional record named `info`, a plain dict written at
save time. Its keys, each written only when the run produced it:
`saved_at` (ISO time, UTC), `corpus` (the corpus directory as given),
`steps`, `batch`, `learning_rate`, `seed`, `final_loss` (the last loss
in the curve), and `held_out_perplexity`. The counted bigram has no
steps and no loss curve; the neural bigram has no held-out perplexity.

The record is assembled in composition, where the settings and the run
result live. A trainer hands its keeper the finished model, as before;
the builder wraps that keeper to add the settings it knows, and the
session saves again once the result exists, so the run numbers join the
record. Both saves happen inside the training process, before anything
crosses the pipe to the chat (ADR-0005).

A file without `info` keeps loading; its record reads as empty.

## Growing the character list on continuation

A checkpoint can seed more training on text that brings characters the
saved model never saw. The character list then grows append-only: every
old character keeps its id, and new characters join at the end, sorted
among themselves. Each trainer accepts a checkpoint as its optional
starting point; it rebuilds the model from the record, grows it to cover
the new text's characters, and trains on. Old weights stay at their old
positions, and the start token's learned row moves with it to the new
last index, so an empty context behaves exactly as before. New
characters begin from fresh, untrained rows. Raw scores for old contexts
stay identical across growth; probabilities over the larger outcome set
renormalize, which is inherent and not a defect. The optimizer's memory
does not travel in the record: a continued run starts with empty
momentum and adaptive-rate state, so its first steps can wobble.

## Consequences

- Old checkpoints load unchanged and keep their generic menu line.
- The checkpoint file is written twice at the end of a run: once as
  soon as the model is done, once when the run numbers are known. The
  second write costs the same as the first.
- A run whose metrics fail keeps its checkpoint, with the record
  missing the numbers the failed metrics would have supplied.
- A continued run's record carries the grown character list, because the
  trainer hands its keeper the finished model with its vocabulary
  already grown.
- The menu reads the whole file to show one line, so a store of large
  checkpoints makes opening the menu slower.
