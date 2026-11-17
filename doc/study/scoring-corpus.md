# Study: the phase 5 scoring corpus

Phase 5 trains a 10-50M parameter model and runs the verdict ADR-0004
sets against GPT-2-small: perplexity per parameter, and grammatical
fluency on the home domain, both measured against a held-out corpus
that both models score. This note names that corpus and decides how
much of it is trained on. How the models are judged is the companion
note [fluency-protocol.md](fluency-protocol.md).

## The corpus

TinyStories is the training corpus for phases 1-5; the choice and its
rationale live in [bigrams.md](bigrams.md). What that note leaves open
is the size and the split. The corpus in the repo today is a
2M-character slice of TinyStories-valid (`data/corpus/tinystories.txt`,
from `scripts/fetch_tinystories.py`). A 10-50M parameter model trained
on 2M characters sees so little text that the run would mostly measure
memorization. A slice that small cannot show what scale does; the
phase 5 run needs the full corpus.

- The phase 5 verdict run trains on the full TinyStories train split
  (about 2.1 million stories), downloaded through the download adapter.
  The network lives only in adapters/composition, as ADR-0002 and
  ADR-0003 require. Nothing dataset-sized is committed.
- The held-out scoring corpus is TinyStories' own valid split (about
  22,000 stories). No model trains on it. The dataset paper itself
  trained on TinyStories-train and validated on TinyStories-valid, so
  the split is the corpus's intended one.
- The 2M slice stays as a smoke-test fixture for quick runs. No verdict
  number comes from it.
- The corpus is fixed for the verdict: it is named here, before the
  training run, and it does not change after results are seen.
- The book corpus (`make corpus`, 37 public-domain books) is not part
  of phase 5. Its question — whether the training text should be
  dialogue-heavy, and what that costs — belongs to the phase 6
  chat-data decisions.
- TinyStories is published under CDLA-Sharing-1.0, which permits use
  and sharing. Nothing in the license blocks training on this machine
  or scoring against it.

## In this repo

- The download adapter fetches both splits into `data/corpus/` and the
  phase 5 run trains on the train split only.
- The scoring harness ([fluency-protocol.md](fluency-protocol.md)) reads
  the valid split and scores both models on the same text.
- Phase 5 verdict numbers come from TinyStories alone. Comparisons on
  other texts are outside the verdict.
