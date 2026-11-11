# ADR-0005: Training progress travels as events

- Status: Accepted
- Date: 2026-11-11

## Context

Training stages run silently: the chat picks a stage and the terminal sits
dark for minutes. The author wants a live line — which step, and how much
time is left. The trainers are plain loops in the application layer; they
must stay screen-blind, and the display must not slow training down.

## Decision

- Trainers emit one event per step through an EventBus port
  (`murrai/application/ports.py`). The event is a `TrainingEvent`: step,
  total_steps, elapsed_ms — self-sufficient, so a listener never asks the
  trainer for anything.
- The bus is optional: trainers take `bus=None` and skip even the clock
  read when nobody listens.
- The concrete bus lives in adapters (`ListenerBus`). Subscribe and
  unsubscribe replace the listener list under a lock; emit iterates without
  a lock (copy-on-write), so emit can never observe a half-registered
  subscriber. Subscribing the same listener twice delivers once; the bus
  matches listeners by equality, since bound methods are never identical
  twice.
- Listener contract: never blocks. The chat's listener only appends to a
  one-slot buffer; a reader thread renders at its own pace and stops when
  the training run ends.

## Consequences

- Trainers gain no knowledge of screens; new progress displays need no
  trainer change.
- A test pins determinism: the same seed produces the identical loss curve
  with and without a listening bus.
- The unsubscribed overhead is one None check per step; the subscribed
  overhead is measured, not asserted.
