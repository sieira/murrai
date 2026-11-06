# AGENTS.md: working rules for this repository

murrAI is a conversational LLM built and trained from scratch on one machine,
as a learning project for the author. Read this fully before starting.
`doc/adr/` holds architecture rationale; `doc/ddd/glossary.md` holds domain
vocabulary. Both bind. `SPEC.md` holds the phase plan and the target.

## Agent roster

| Agent | Owns | Never does |
|-------|------|------------|
| pm | phases, backlog order, session plans, ADR drafting, GitHub issues, README coherence | code, tests |
| builder | `murrai/` code, `tests/`, `Makefile`, tooling, CI config | scope expansion, weakened tests |

## Hard rules

1. **Author understanding gate.** Agents never implement what the author has
   not clearly understood. Doubt stops work and becomes a direct question, in
   plain language, carrying the context needed to answer it, through the
   opencode question tool, one question at a time; the run waits for the
   answer. Scope expansion is a proposal back to the main agent, never an
   action. An agent going deep without facts stops and asks.

2. **TDD is enforced.** Every deterministic change ships with a failing test
   written first, shown red, then green. A test earns its name by being able
   to fail. Training runs are experiments, not code behavior: loss curves,
   sample outputs, and timings are their evidence. Iterate with
   `make test-one SCRIPT=<substring>`; `make check` runs once, at session
   end, by the main agent.

3. **ADRs.** Structural decisions live in `doc/adr/`: numbered files
   (`NNNN-kebab-case-title.md`), Nygard format (Status / Date / Context /
   Decision / Consequences). ADRs are living records: they evolve by
   editing, and they do not need to grow in number or size; a new file is
   for a genuinely new decision. An ADR edit enforces the decision in the
   artifact in the same increment: the change that edits the ADR also edits
   the code, config, or docs implementing that decision. Structural changes
   need an ADR edit or the author's approval; small implementation choices
   do not (ADR-0001).

4. **Deletion first.** A goal met by deleting beats one met by editing;
   editing beats adding. Docs, comments, and reports carry no replaced
   mechanisms, no receipts of removed things, no discussion of what is not,
   unless it adds something.

5. **Plain language.** Everything addressed to the author (reports, plans,
   questions, summaries) speaks plainly: no jargon, no invented compound
   terms, no acronyms. Where the repo's own term is the precise word, use it
   and explain it in the same sentence.

6. **Unslop, systematic.** Every author-facing artifact (questions, reports,
   plans, README, ADRs, glossary entries, PR bodies) gets an unslop pass
   before it reaches the author or the repo (skill `unslop`). It covers
   prose; code cleanliness rests on rule 7 and review.

7. **Comments.** Default no comment. A comment earns its line only for a
   non-obvious why. Banned: narration, summaries of adjacent code,
   design-history essays.

8. **Git discipline.** One session = one worktree = one branch
   (`../murrai-<topic>`); subagents share it. The main folder stays on main.
   Commit by pathspec. Before every commit run `git branch --show-current`;
   wrong branch means abort and report. Everything lands via PR and
   squash-merges; PR titles follow Conventional Commits
   (`<type>(<scope>)!: <subject>`); a PR resolving an issue carries
   `Fixes #N`. A PR over ~30 files splits or stops. PRs are created and
   merged only with the author's explicit approval — a PR may be pushed
   and its review summary reported, but `gh pr merge` (or any equivalent)
   waits for the author's go.

9. **Evidence over claims.** "It works" needs evidence: a test run, a loss
   curve, a sample transcript. A subagent report is a claim; verify it
   against the tree before acting on it.

10. **Machine boundaries.** `make boundaries` enforces the hexagon
    (ADR-0003). Extend the checks; never bypass them.

11. **Session ends with a delivered result.** It ends at a pushed PR, or at
    an explicit author-abort with a written handoff. Never leave stranded
    work.

12. **Board discipline.** Work follows the "murrAI — work board"
    (https://github.com/users/sieira/projects/2): every session's work ties
    to an issue on the board, the issue carries a P0-P3 priority, and the
    row is removed when the work lands.

## Commands

```bash
make setup                     # .venv + dependencies
make check                     # everything CI runs; session-end gate
make test                      # full pytest suite
make test-one SCRIPT=<substr>  # one test while iterating
make lint                      # ruff
make boundaries                # import-linter layer check (ADR-0003)
```

`train` and `chat` targets join the Makefile when phases 5 and 6 build them.

## Definition of done

- A failing test was shown before the change (red), then green.
- Every increment ships a runnable example of the result (demo, script,
  or command) alongside the tests.
- `make check` passes locally; the main agent runs it once at session end.
- Claims carry evidence: test output, loss curve, or sample transcript.
- Domain tests run with no network and no GPU.
- Rule 1 holds: the author understood every implemented concept first.

## Merge and versioning policy

All work lands via PR and squash-merges. Before 1.0, features bump the minor
version and fixes the patch. Tagging is deliberate and happens when the
author asks.
