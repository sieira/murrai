# ADR-0005: Training reports progress as events, from its own process

- Status: Accepted
- Date: 2026-11-11

## Context

Training takes minutes and is pure Python, so it keeps the interpreter
busy. The interface must show live progress and stay responsive while a
training runs. Trainers must stay unaware of screens, and reporting must
not change training results or slow training down.

The handoff of a finished model between processes is unreliable: it dies
while the receiving side unpacks it, about three runs in ten. The
checkpoint save ran after that handoff, so a training that died there
left nothing on disk.

## Decision

- Trainers emit one `TrainingEvent` (step, total_steps) per step through
  an optional `EventBus` port (`murrai/application/ports.py`). A reader
  never asks the trainer for anything; it derives pace and time left
  from its own clock. Without a bus, a trainer does nothing extra.
- `ListenerBus` (`murrai/adapters/events.py`) is the concrete bus.
  Subscribing replaces the listener list under a lock, emitting walks it
  without one, and a listener subscribed twice is delivered once. A
  listener never blocks.
- A training runs in its own process, started with `spawn`. The child
  sends `("step", step, total)`, then `("done", name)`, where name is
  the checkpoint the training saved under, or `("failed", reason)` over
  a pipe; a thread in the chat turns steps into calls on the status
  registry and hands the name to the screen's event loop, where a `load`
  function turns it into the session that replaces the chat's. A child
  that ends without a result counts as failed. The child's standard
  output and error are silenced, because the chat owns the terminal. A
  command that trains is therefore picklable (a `functools.partial` of a
  module-level function, in composition) and saves its checkpoint before
  anything else runs after the last step.
- Only the checkpoint name crosses the process boundary; the trained
  model never does. The chat's `load` turns the name into a session by
  reading the checkpoint store, the same read its menu load commands
  use, so training and loading use the same code path.
- The status registry (`murrai/presentation/chat/status.py`) holds one
  entry per running training. The status bar renders one bar per entry
  with its time left; a failed entry stays visible as "name failed". The
  registry wakes the screen at most ten times a second, and always when a
  run ends.
- The counted bigram counts the corpus in chunks and reports each chunk;
  its step unit is characters.

## Consequences

- Trainers know nothing about screens; a new progress display changes no
  trainer.
- Trainers know nothing about files either: a trainer calls a `keeper`
  function it is handed with the finished model, and only the wiring in
  composition connects that function to the checkpoint store.
- The same seed produces the identical loss curve with and without a bus;
  a test pins this.
- The interface and the trainer do not share an interpreter, so training
  load cannot delay a keystroke.
- A training command cannot close over local state; whatever it needs
  travels as picklable data.
- The pipe carries only a name, and the checkpoint is already on disk,
  so the handoff cannot lose a finished training.
