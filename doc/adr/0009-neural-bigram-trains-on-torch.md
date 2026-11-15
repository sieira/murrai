# ADR-0009: The neural bigram trains on torch tensors

- Status: Accepted
- Date: 2026-11-15

## Context

The neural bigram was the first model trained through the hand-written
scalar autograd engine, built in the foundations phase. After the
byte-level BPE tokenizer widened its output from 117 to 556 classes,
one training step walks a 557 by 556 weight grid in Python and builds
a graph of roughly 70,000 scalar nodes. Measured: 0.4 seconds per
step, so the 1000-step convergence test alone runs 6.7 minutes and
the full test gate takes 8. Every model trained after it already uses
torch, where the same step costs well under a millisecond.

## Decision

- The `NeuralBigram` stores its weights as one torch tensor: one row
  per previous token, one column per next token, and an extra row for
  the start token. `batch_loss` is `cross_entropy` over the gathered
  rows; gradients flow through torch; the per-step update is one tensor
  operation. The trainer loop keeps its shape: sample a batch, compute
  the loss, step backwards, update.
- Checkpoints keep the nested-list weight format, so BPE-era saves load
  unchanged.
- The hand-written engine (`murrai/domain/autograd.py`) leaves the
  training path. It stays as its own tested artifact with its demo
  script; it is the study piece of the foundations phase, and no
  trainer depends on it.

## Consequences

- The convergence test keeps its 1000 steps and its 15 percent bar and
  runs in 0.6 seconds instead of 6.7 minutes; the full test gate runs
  in well under a minute.
- The training is identical in shape, but the loss values now carry
  float32 rounding instead of Python float precision.
- New models train on torch from birth; the scalar engine is no longer
  a dependency of any trainer.
