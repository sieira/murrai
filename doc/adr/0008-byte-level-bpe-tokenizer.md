# ADR-0008: Text is tokenized by byte-level BPE

- Status: Accepted
- Date: 2026-11-13

## Context

The model reads ids from a fixed table, so text becomes ids first. The
table must be complete. Every string the user can type encodes, whatever
the training corpus held. The id sequences must also be as short as the
corpus allows. The context window holds meaning, not padding.

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
  sets the vocabulary size; training runs default to 300.

## Consequences

- Measured at 300 merges (scripts/tokenizer_density.py): English prose
  encodes 3.2x denser than one token per character; unseen text 2.1x.
- Fit and encode no longer do a full slow pass over the text for every
  merge: the pair counts update only around each occurrence, and the text
  is stored as 4-byte numbers so searching and copying run at memory
  speed. Measured on the 26.5M-byte corpus at 300 merges
  (scripts/tokenizer_fit_speed.py): fit ~70s, encode ~62s. Both still
  make one fast pass per merge over the shrinking sequence; a
  neighbour-chain layout (each piece knowing only its left and right
  neighbour, with a memory of where each pair lives) would remove those
  passes for about 2GB of memory; revisit if this scale stops being
  comfortable.
- A single token may hold bytes that are not valid UTF-8 on their own;
  decoding and display render them with replacement characters.
