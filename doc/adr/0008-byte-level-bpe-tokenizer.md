# ADR-0008: A byte-level BPE tokenizer replaces the char vocabulary

- Status: Accepted
- Date: 2026-11-13

## Context

Phases 2 and 3 built the vocabulary from the corpus: one id per distinct
character. Two failures follow. Text outside the corpus cannot be encoded
at all — an unknown character stops the run. And one token per character
is the longest possible encoding, spending the context window on nothing.
Phase 4 (issue #6) replaces this with a tokenizer learned from the corpus,
in the minbpe style. That is also where GPT-2, our ADR-0004 target, sits.

## Decision

- The char vocabulary is replaced by a byte-level BPE tokenizer, the
  `TokenizerEncoderDecoder` domain type. The base vocabulary is the 256
  byte values, so every string encodes; nothing is unknown.
- Training repeatedly merges the most frequent adjacent pair (ties broken
  deterministically). The ordered merge table is the learned vocabulary;
  its size is 256 plus the number of merges.
- Encoding replays the merges in the order they were learned. Decoding
  concatenates each id's byte sequence, so decoding an encoding returns
  the original string, for any string.
- Training, encoding, and decoding are separate operations.
- Models consume the tokenizer. Checkpoints store the merge table instead
  of the character list. Old char-based checkpoints become unreadable;
  accepted, the project is pre-1.0.
- The number of merges is the one knob. Training runs default to 300.

## Consequences

- Measured at 300 merges on the demo corpus
  (scripts/tokenizer_density.py): 3.2x denser than char-level on the
  training corpus, 2.1x on unseen text.
- Encoding cost grows with the number of merges times the sequence
  length. Fine at this scale; revisited before fitting thousands of
  merges on the full corpus.
- The chat must render single-token bytes that are not valid UTF-8 on
  their own; how is decided in code.
- The char vocabulary class is deleted, not deprecated.
