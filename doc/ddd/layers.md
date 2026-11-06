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
- `adapters/` implement `application/` ports; they are the only layer that
  touches the network or the filesystem.
- `composition/` is the only layer that names concrete adapters.
