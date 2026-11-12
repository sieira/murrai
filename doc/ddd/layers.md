# Layers

The dependency map enforced by import-linter (ADR-0003). An arrow reads
"may import". Anything not listed is forbidden.

```
composition   → presentation, adapters, application
presentation  → application
adapters      → application
application   → domain
domain        → (nothing above)
tests         → anything in murrai/
```

Notes:

- `domain/` is pure math: torch tensor operations are allowed; torch I/O
  (save/load), network access, and filesystem writes are not (ADR-0002).
- `presentation/` renders state and samples; it computes no math.
- The chat (screen app, command panel, chat session) is its own separable domain:
  it lives in `presentation/chat/` plus its composition factories and
  touches the core only through the `LanguageModel` port, so it could be
  extracted into its own project later. Nothing above presentation depends
  on it.
- `adapters/` implement `application/` ports; they are the only layer that
  touches the network or the filesystem.
- `composition/` is the only layer that names concrete adapters.
