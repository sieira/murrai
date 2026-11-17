# Study: the phase 5 fluency protocol

ADR-0004 sets two tests for the phase 5 verdict. First, perplexity per
parameter against GPT-2-small on a held-out corpus, named in
[scoring-corpus.md](scoring-corpus.md). Second, grammatical fluency on
the home domain, judged by the author on side-by-side samples. This
note fixes the method for both before the overnight MPS run, because a
method chosen after seeing the results measures nothing.

## The scoring harness

- Both models score the same text: the TinyStories valid split
  ([scoring-corpus.md](scoring-corpus.md)). GPT-2-small enters through
  an adapter behind the existing ports, like any other model.
- The two models use different tokenizers (murrAI's own BPE, ADR-0008;
  GPT-2's own), so perplexity per token does not compare. The harness
  converts each model's held-out loss to perplexity per byte, then
  divides by parameter count. Lower wins. murrAI passes if its number
  is at or below GPT-2-small's.
- A fail is an acceptable recorded outcome. The verdict carries both
  models' numbers; a claim without the measurement is not a verdict.

## The fluency protocol

- A fixed set of 20 prompts, taken from story openings in the held-out
  valid split. The dataset also publishes the evaluation prompts used
  in the paper; switching to those is an edit to this note, not a
  silent change.
- Each model generates one sample per prompt, with the same sampling
  settings and a fixed seed.
- The transcripts are labeled A and B in random order. The author reads
  them blind and judges each sample on one question: is it grammatical
  English, sentence after sentence? The labels are revealed after the
  judging.
- Pass: murrAI's rate of grammatical samples is equal to or higher than
  GPT-2-small's across the 20 prompts. Fail: lower. The verdict, the
  transcripts, and the numbers are recorded either way. Whether the
  samples tell a coherent story is part of the overall ADR-0004 claim;
  the fluency gate measures grammar.

## The gate

- Both this note and [scoring-corpus.md](scoring-corpus.md) are accepted
  by the author before the overnight MPS training run starts. No
  compute is spent on an undecided protocol.

## In this repo

- The phase 5 harness must produce, for each model: held-out loss as
  perplexity per byte, the parameter count, one sample per prompt, and
  a transcript file the author reads.
- The prompt set, the sampling settings, and the seed freeze when the
  run starts, as the corpus does. If a protocol flaw shows up mid-run,
  the fix is recorded by editing this note before any verdict is
  claimed.
