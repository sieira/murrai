# ADR-0006: The chat owns the whole screen

- Status: Accepted
- Date: 2026-11-12

## Motivation

The chat lived on a prompt line. Pressing Enter jumped or duplicated the
typed line, and the ghost shadow only existed while the prompt was up.
Switching a model left the chat for a separate palette screen, and a
training run took over the whole terminal until it finished.

## Decision

- The chat is one full-screen `Application` that owns the terminal until
  the user quits with ctrl-c. The cursor is a block.
- The story pane at the top mirrors the session. Typed text is plain,
  generated text is cyan. It shows the tail of the story.
- The write pane at the bottom holds the live line. The model's offered
  shadow sits after the cursor. Tab takes the whole offered shadow.
  Enter commits the typed text alone and never takes the shadow.
- ctrl-p opens a command panel on the right. The panel filters as you
  type, moves with the arrow keys, and rebuilds its list every time it
  opens. Escape or ctrl-p closes it.
- Picking a load command swaps the session in place; the story and the
  live line follow the new model. Picking a training command starts it
  in the background. A status bar at the bottom shows one bar per
  running training (ADR-0005). A finished training is saved as a
  checkpoint and picked up by hand through the panel.
- A chat with no trained model starts with the panel open, over an empty
  story.

## Consequences

- prompt_toolkit owns every component with a cursor in it (the live
  line, the panel, the shadow); the chat never leaves the screen.
- The tests drive the app with piped keystrokes, so the interface runs
  with no terminal attached.
- The palette screen, the standalone picker prompt, and the separate
  progress bar are gone; commands and training live inside the chat.
