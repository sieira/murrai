# ADR-0007: The chat explains itself through log lanes

- Status: Accepted
- Date: 2026-11-12

## Context

Finding why the chat froze during training took throwaway probe scripts
that patched the program from outside. Each question about the screen
needed a new probe. The chat should answer "what happened, and when"
from its own record.

## Decision

- The chat publishes facts as immutable events (`ChatEvent`,
  `murrai/presentation/chat/lanes.py`). An event is plain data and renders
  its own lane line. Lanes: `[panel]` opened and closed, `[command]`
  picked, `[session]` swapped, `[train]` started, stepped, finished,
  failed (with the reason).
- One observer port (`EventObserver.on_event`, silent by default) and one
  subscriber, `ChatLog`. It stamps each lane line with a monotonic clock,
  keeps the lines in a list (the test seam) and appends them to a file
  when given a path. `scripts/chat.py` writes `chat.log`, which git
  ignores. Nothing prints to the terminal, because prompt_toolkit owns it.
- Publishers (the command panel, the training status, `run_chat`) take the
  observer as an optional argument. Composition wires it. Trainers stay
  unaware of it; they keep emitting `TrainingEvent` (ADR-0005), and the
  status turns those into `[train]` lines on the parent side.

## Consequences

- A freeze, a failed run, or a slow swap is read from `chat.log`
  timestamps, with no probe.
- A new fact is a new event class with a lane line; no new plumbing.
