# ADR-0006: The chat is one full-screen application

- Status: Accepted
- Date: 2026-11-12

## Context

A chat needs live editing of one line, a model's offered continuation, a
way to switch models, and visible background work, all at once. Separate
prompts and screens for each cannot share state or stay on screen
together.

## Decision

- The chat is one full-screen prompt_toolkit `Application` that owns the
  terminal until the user quits with ctrl-c. Every component with a
  cursor lives inside it. The cursor is a block, drawn green.
- The story pane mirrors the session: typed text plain, generated text
  cyan, showing the tail. The write pane holds the live line with the
  model's offered ghost after the cursor. Tab takes the whole ghost;
  Enter commits the typed text alone and never takes it.
- ctrl-p and f9 both focus the menu bar from the write pane; escape
  returns to the write pane. The bar toggles, so one key walks back out.
- Picking a load command swaps the session in place. Picking a training
  command starts it in the background (ADR-0005); a finished training
  saves its checkpoint and its built session replaces the chat's when it
  lands. A chat with no trained model opens with the menu focused.
- A command is one selectable action. Composition builds the list of
  commands; presentation only shows them.
- A menu bar sits on the top line, its dropdowns floating over the
  story like the form. Composition builds the menu content: it groups the
  commands it already supplies into new, load, and retrain; presentation
  renders them as menus, items and submenus.
- Mouse reporting is on, so clicks reach the screen. In most terminals,
  selecting text then needs Option (macOS) or Shift.

## Consequences

- The interface runs under tests with piped keystrokes and no terminal.
- A new action is a new command in composition; the screen does not
  change.
- Anything slow that a command does on the main thread freezes the whole
  screen, because the screen has one thread of its own.
