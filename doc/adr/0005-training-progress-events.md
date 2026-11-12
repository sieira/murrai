# ADR-0005: Training reports progress as events, from its own process

- Status: Accepted
- Date: 2026-11-11

## Context

Training takes minutes and is pure Python, so it keeps the interpreter
busy. The interface must show live progress and stay responsive while a
training runs. Trainers must stay unaware of screens, and reporting must
not change training results or slow training down.

## Decision

- Trainers emit one `TrainingEvent` (step, total_steps) per step through
  an optional `EventBus` port (`murrai/application/ports.py`). The event
  is self-sufficient; a reader never asks the trainer for anything and
  derives pace and time left from its own clock. Without a bus, a trainer
  does nothing extra.
- `ListenerBus` (`murrai/adapters/events.py`) is the concrete bus.
  Subscribing replaces the listener list under a lock, emitting walks it
  without one, and a listener subscribed twice is delivered once. A
  listener never blocks.
- A training runs in its own process, started with `spawn`. The child
  sends `("step", step, total)`, then `("done",)` or `("failed", reason)`
  over a pipe; a thread in the chat turns them into calls on the status
  registry. A child that ends without a result counts as failed. The
  child's standard output and error are silenced, because the chat owns
  the terminal. A command that trains is therefore picklable (a
  `functools.partial` of a module-level function, in composition) and
  saves its own checkpoint.
- The status registry (`murrai/presentation/chat/status.py`) holds one
  entry per running training. The status bar renders one bar per entry
  with its time left; a failed entry stays visible as "name failed". The
  registry wakes the screen at most ten times a second, and always when a
  run ends.

## Consequences

- Trainers know nothing about screens; a new progress display changes no
  trainer.
- The same seed produces the identical loss curve with and without a bus;
  a test pins this.
- The interface and the trainer do not share an interpreter, so training
  load cannot delay a keystroke.
- A training command cannot close over local state; whatever it needs
  travels as picklable data.
- Anything that must return an object to the chat (a built session) needs
  a channel beyond this protocol, which carries progress only.
