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
- ctrl-p opens a command panel on the right: a list of command names,
  moved with the arrow keys, with the highlighted command's description
  in a box below. The list is rebuilt each time the panel opens. Escape
  or ctrl-p closes it. The panel and the write pane share one vertical
  split, so their boxes align.
- Picking a load command swaps the session in place. Picking a training
  command starts it in the background (ADR-0005); a finished training
  saves its checkpoint and its built session replaces the chat's when it
  lands. A chat with no trained model opens with the panel open.
- A command is one selectable action. Composition builds the list of
  commands; presentation only shows them.
- A menu bar sits on the top line: `models`, `sessions`, and `about`,
  each a menu whose items open a dropdown under the bar. The bar is
  always on screen. Opening a menu selects its first item; the arrow
  keys move between menus and within the open menu, wrapping; enter or
  space picks the selected item; escape closes the menu and returns to
  the editor; a letter picks the first item starting with it, or opens
  the first menu whose title starts with it. Mouse clicks work: a click
  on a title opens or closes that menu, a click on an item picks it.
  While a menu is open, the selected item's description shows on the
  bottom status line before the run info. Alt-m, alt-s, and alt-a open
  the three menus. The models and sessions menus are empty for now and
  fill in later issues. The about item opens a plain message dialog
  with one line about the project; its version and license content is
  a later issue.

## Consequences

- The interface runs under tests with piped keystrokes and no terminal.
- A new action is a new command in composition; the screen does not
  change.
- Anything slow that a command does on the main thread freezes the whole
  screen, because the screen has one thread of its own.
