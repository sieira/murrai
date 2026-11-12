# ADR-0005: Training progress travels as events

- Status: Accepted
- Date: 2026-11-11

## Motivation

Training stages used to run silently. Picking one left the terminal dark
for minutes. The author asked for a live line that names the current step
and the time left. Two constraints shaped the answer. Trainers stay
screen-blind, and the display must not slow training down.

## Decision

- Trainers emit one event per step through an `EventBus` port
  (`murrai/application/ports.py`). The event is a `TrainingEvent` and
  carries step and total_steps. It is self-sufficient; its reader never
  asks the trainer for anything. The reader computes pace and remaining
  time from its own clock.
- The bus is optional. Trainers take `bus=None` and do nothing when no bus
  was given.
- The concrete bus is `ListenerBus` (`murrai/adapters/events.py`).
  Subscribe and unsubscribe replace the listener list under a lock. Emit
  walks the list without a lock, so a listener is never seen
  half-registered. Subscribing one listener twice delivers once; the bus
  compares listeners by equality because bound methods are never identical
  twice.
- A listener never blocks. The chat's listener records the event in a
  status registry. The chat's status bar renders one bar per running
  training, and the redraw hook wakes the screen at most ten times a
  second, so a fast trainer cannot flood the interface; a run ending
  always wakes. A training runs off the
  main thread, so the chat stays usable while it trains. A finished run
  is saved as a checkpoint and leaves the status bar; a failed run stays
  visible as "name failed".

## Consequences

- Trainers know nothing about screens. A new progress display changes no
  trainer.
- The same seed produces the identical loss curve with and without a
  listening bus; a test pins this.
- Overhead without a subscriber is one None check per step. The subscribed
  overhead is measured, not asserted.
- Rendering belongs to the chat's status bar; the adapter owns only the
  mapping from event to bar line.
