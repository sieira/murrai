# ADR-0010: What a session save holds

- Status: Accepted
- Date: 2026-11-17

## Context

A chat story dies when the app quits. The typed and generated text
lives only in the running session, and nothing can write it down yet.
The sessions menu saves a story with one command and loads it with
another, and it lists one load command per saved story.

## Decision

A session save holds three things:

1. the story's token ids, in order,
2. one origin per id, `entered` or `generated`, which are the colors,
3. the name of the checkpoint the session ran on.

A load rebuilds the session from the save: the model comes back by
its saved name, and the loaded session rolls with a fresh seed. The
store keeps one JSON file per save, named after the save, and the
file adapter writes it atomically. The author names each save in a
one-field form.

The port is `StoryStore` in application: `save`, `load`, and `names`.
The saved shape is a frozen dataclass, `StorySave`. The adapter is
`FileStoryStore` in adapters. The sessions menu gets a "save story"
command, which opens the one-field form, and one "load *name*" command
per save.

## Consequences

- Quitting and reloading reproduces the story exactly, ids and
  origins together.
- Everything a loaded session generates runs on a fresh roll.
- The store grows by JSON only.
