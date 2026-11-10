# Study: the MLP language model

The theory behind phase 2: how a fixed window of characters enters a
network, and what one training step does when torch owns the engine.
Plain language throughout; the code and the glossary use the same words.

## The job

The bigram's whole memory is one character. Real text carries more: the
char before it changes the odds, and the one before that. Phase 2 grows
the memory to a fixed window of k previous characters and feeds it
through an MLP — a lookup, a hidden layer, an output layer — while
keeping everything else about training the same.

## Embeddings are lookup tables

The neural bigram already holds one learned row per character:
`weights[prev_id]` picks a row by index, and training nudges every
number in it. That structure — one learned row per token, fetched by
indexing — is exactly what an embedding table is. Two things change:

- The row is shorter than the vocabulary (a 117-char vocabulary, rows
  of 16 numbers) and describes the token rather than predicting: 16
  learned numbers saying what 't' is like for the task.
- The row feeds further layers; the prediction moves to the network's
  output end.

The textbook definition, one-hot times matrix, is the same machine:
multiplying a matrix by a one-hot selects exactly one row, and indexing
does it without building the one-hot.

## Concatenation is how the window enters

Each of the k window characters looks up its own vector. Combining them
into one input has one right answer here: lay them side by side. Each
position keeps a private slice of the long vector, and the hidden
layer's weights are position-specific — the first char matters through
its slots, the last char through others, and 'th' never collapses into
'ht'. Summing or averaging would erase position and order at the input,
and no training can recover information thrown away there. A subtler
win: the same character is treated differently at different positions,
because different weight slices touch its vector at each slot.

## One training step, torch edition

The loop from the micrograd notes survives unchanged; only the engine
swaps. Parameters become torch tensors marked `requires_grad=True` (the
Value flag in tensor form); `nn.Module` is the container that lists them
for the update. `loss.backward()` is the walk — same chain rule, one
sweep, filling `.grad` everywhere, embedding rows included. Rows the
batch never looked up get zero gradient and sit the step out. The nudge
stays `w <- w - rate * gradient(w)`, which `optimizer.step()` performs
as bookkeeping; `zero_grad()` is the reset. The loss is the same
negative log-likelihood, renamed cross-entropy and fused into one
overflow-safe function: torch subtracts each row's maximum before
exponentiating, the cap the `_exp` helper did by hand.

## What was built

- `murrai/domain/mlp.py`: `MlpLM` — embedding table (one extra start
  row, left-padding short histories), concatenation, hidden layer with
  tanh, output layer, softmax; `nll`/`perplexity` for held-out text;
  seeded generation.
- `murrai/application/runs.py`: `train_mlp_lm` (minibatch cross-entropy
  over sliding windows) and `plateau_rate` — when the loss stalls for
  `patience` steps, the learning rate shrinks.
- `murrai/composition/chat.py`: the "windowed mlp" stage, k=4.
- `scripts/train_mlp_demo.py`: the evidence runner.

## Evidence (Aesop, 52,746 train chars, 5,861 held out)

The counted-bigram baseline on the same split: held-out perplexity
12.866. The windowed MLP (window 4, 16 dims, hidden 128, 5,000 steps,
batch 64, rate 0.5 with plateau shrink, seed 0): **8.997** — a 30%
reduction, and the gate for the phase. Train loss fell 4.764 -> 1.577.

Context-window ablation (same config, one knob turned; held-out
perplexity):

| window | 1      | 2      | 4     | 8     |
|--------|--------|--------|-------|-------|
| held-out perplexity | 13.383 | 10.174 | **8.997** | 9.859 |

Wider windows buy real accuracy up to four characters — one char of
memory barely matches the counted table — and then give some back at
eight: more parameters over the same 53k characters stops paying. The
window's worth is bounded by the corpus that feeds it.

The learning rate decayed to zero by the end of the run (several
plateau shrinks); late steps cost nothing but change nothing either.

Sample at seed 17 reads like roughed-up English next to the bigram's
character salad — word shapes, sentence starts, punctuation habits:

```
, Mouk to the Ass
offer pur shad. The Croved to eingustson hive moul
 The Surier
 The Hrotes AplaS repuries tperted had.
```

## What it unlocks

Phase 3 stacks self-attention on top of this skeleton: the window stops
being a fixed k and the model learns which positions to read. The
training loop, the loss, and the schedule carry over untouched.
