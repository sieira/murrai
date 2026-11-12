# ADR-0007: The chat records itself as events in log lanes

- Status: Accepted
- Date: 2026-11-12

## Context

The chat runs on a terminal it owns, so printing is impossible and
attaching from outside is fragile. Questions about what the interface did
and when must be answerable from the interface's own record.

## Decision

- The chat publishes facts as immutable events (`ChatEvent`,
  `murrai/presentation/chat/lanes.py`). An event is plain data and renders
  its own lane line. Lanes: `[panel]` opened and closed, `[command]`
  picked, `[session]` swapped, `[train]` started, stepped, finished,
  failed (with the reason).
- One observer port, `EventObserver.on_event`, silent by default, and one
  subscriber, `ChatLog`. It stamps each lane line with a monotonic clock,
  keeps the lines in a list, and appends them to a file when given a path.
  `scripts/chat.py` writes `chat.log`, which git ignores.
- Publishers (the command panel, the training status, `run_chat`) take
  the observer as an optional argument; composition wires it. Trainers do
  not publish chat events; the status turns their `TrainingEvent`s into
  `[train]` lines.

## Consequences

- A freeze, a failed run, or a slow swap shows in the log's timestamps.
- A new fact is a new event class with a lane line; no new plumbing.
- Events carry ids and values only, never the story text.
