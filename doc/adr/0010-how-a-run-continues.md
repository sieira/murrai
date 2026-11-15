# ADR-0010: How a run continues

- Status: Accepted
- Date: 2026-11-15

## Context

A checkpoint holds a trained model and its tokenizer (ADR-0009). We
want to keep training on from a checkpoint instead of starting over:
the user spends a run's time once and improves the same model later.

## Decision

Each gradient trainer accepts a checkpoint as its starting point. The
checkpoint's tokenizer is used unchanged: the new text is encoded
through the saved merge list, and no new merging is fitted. The weights
rebuild from the checkpoint and train on.

The learning rate is whatever the new run says: the optimizer's memory
starts empty and the plateau schedule restarts. A call that passes
shape settings (window, dimension, hidden, depth, max_time, heads)
that contradict the checkpoint is rejected; the checkpoint keeps its
own shape.

The counted bigram has no gradient steps, so it does not continue; it
always counts from scratch.

## Consequences

- A continued run's loss curve starts near the checkpoint's final
  loss, not from scratch.
- The chat form omits the shape knobs when a start from a checkpoint
  is chosen, so the rejection above is a guard, not the usual path.
