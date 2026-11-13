# ADR-0008: Text is tokenized by byte-level BPE

- Status: Accepted
- Date: 2026-11-13

## Context

The model reads ids from a fixed table, so text becomes ids first. The
table must be complete: every string the user can type encodes, whatever
the training corpus held. And the id sequences must be as short as the
corpus allows: the context window holds meaning, not padding.

## Decision

- The `TokenizerEncoderDecoder` (murrai/domain/tokenizer.py) turns text
  into ids and back. Its base vocabulary is the 256 byte values, so every
  string encodes.
- `fit` learns an ordered merge table: it repeatedly merges the most
  frequent adjacent pair (ties broken deterministically) into a new id.
  The vocabulary is the 256 bytes plus one id per merge.
- `encode` turns text into UTF-8 bytes and replays the merges in the
  order they were learned. `decode` concatenates each id's byte sequence.
  Decoding an encoding returns the original string, for any string.
- Training, encoding, and decoding are separate operations. Models take
  the tokenizer; checkpoints store the merge table. The number of merges
  is the vocabulary-size knob; training runs default to 300.

## Consequences

- Measured at 300 merges (scripts/tokenizer_density.py): English prose
  encodes 3.2x denser than one-token-per-character; unseen text 2.1x.
- Encoding cost grows with merges times sequence length; fine at this
  scale, revisited before fitting thousands of merges on the full corpus.
- A single token may hold bytes that are not valid UTF-8 on their own;
  decoding and display render them with replacement characters.
