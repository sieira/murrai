# ADR-0009: What a checkpoint holds

- Status: Accepted
- Date: 2026-11-13

## Context

A checkpoint holds a model kind, the tokenizer's merge list, and the
weights (ADR-0008). We want a checkpoint to record when it was
trained, on which corpus, with which settings, and how well the model
did. The chat menu shows a load command for every checkpoint, and we
want each command to carry those facts instead of one generic line.

## Decision

A checkpoint holds four things: the model kind, the tokenizer's merge
list, the weights, and a record of its run. The record is a frozen
dataclass, `CheckpointRecord`. The store keeps it as a small JSON file
beside the weights file, so reading the record never opens the
weights. A checkpoint's name carries its kind and the text it trained
on (`neural-bigram-philosophy`), so runs on different texts keep their
own checkpoints. The record carries the run's settings (`saved_at` as
ISO time in UTC, `corpus`, `seed`, and `corpus_chars`, the characters
of text the run trained on, written when a size cap was applied), and,
when the run produced them, a gradient run (`steps`, `batch`,
`learning_rate`, and `final_loss`, the last loss in the curve, written
once the curve is known) and `held_out_perplexity`. The counted bigram has no gradient run; the
neural bigram has no held-out perplexity. A run that holds out no text
reports no perplexity. A part the run did not produce is absent.
Composition assembles the record because it holds the settings and the
run result. A trainer hands its keeper the finished model, as before;
the builder wraps that keeper to add the settings it knows, and the
session writes the record again once the result exists, so the run
numbers join it. Both writes happen inside the training process,
before anything crosses the pipe to the chat (ADR-0005).

A checkpoint is also a starting point. A gradient trainer may take one
as its starting point, rebuild the weights, and keep training. The
tokenizer travels unchanged, and the new run fits no merge list. The
learning rate and the plateau schedule come from the new run. The
trainer rejects a call whose shape settings contradict the checkpoint;
shape settings left out leave the shape to the checkpoint. The counted
bigram has no gradient steps, so it always counts from scratch.

## Consequences

- The weights file is written once. The record file is written twice
  at the end of a run: once with the settings, once when the run
  numbers are known.
- A run whose metrics fail keeps its checkpoint. The record keeps the
  settings and misses the run numbers.
