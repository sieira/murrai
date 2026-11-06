# ADR-0004: Target: match or beat GPT-2-small per parameter

- Status: Accepted
- Date: 2026-11-06

## Context

The author asked whether a clean-slate build can beat early-boom models.
Published evidence says yes, per parameter and per compute:

- TinyStories (2023): 1-10M parameter models trained on one clean, curated
  corpus produced more coherent, more grammatical English than GPT-2-XL
  (1.5B parameters, 2019), by human evaluation.
- SmolLM2-135M (2024) beats GPT-2-124M across evaluations; the difference
  is data curation and modern training recipes, not scale.
- GPT-2 speedrun reproductions (2024) reach GPT-2-124M quality about 8x
  cheaper than the original recipe, showing the old recipe carried slack a
  clean rebuild harvests.

Scale remains physics: GPT-2-small saw roughly 40B tokens; murrAI will see
1-2B on this machine. Within-domain coherence is reachable; world knowledge
and general chat are not.

## Decision

murrAI trains a 10-50M parameter model on this machine (Apple M4, MPS) and
targets, at Phase 5:

1. Perplexity per parameter at or better than GPT-2-small on a held-out
   corpus that both models score. The corpus is named in a Phase 5 ADR.
2. Grammatical fluency on murrAI's home domain, judged by the author on
   side-by-side samples against GPT-2-small output.

The verdict is recorded pass or fail with the evidence. A recorded failure
is an acceptable outcome; an unmeasured claim is not.

## Scope limits

- Within-domain coherence only. World knowledge and general chat are out of
  scope.
- Chat output is expected to remain crude; that is by design, not a defect.

## Consequences

- Phase 5 includes a scoring harness that runs GPT-2-small on the same
  corpus (an adapter behind the existing ports).
- The corpus choice and the fluency protocol get their own Phase 5 ADRs.
