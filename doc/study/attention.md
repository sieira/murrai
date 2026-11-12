# Study: attention

Attention is a network's mechanism for deciding, per prediction, which
earlier positions to read and how much to weigh each. This note covers
the theory behind phase 3, where the fixed window of characters is
replaced by that decision.

## The job

The windowed mlp (multi-layer perceptron) reads a fixed slice of the
past: the last 8 characters, always the same slice, always in the same
slots. Two things
it cannot do: look further back when it matters, and treat the same
character differently depending on what surrounds it. Phase 3 replaces
the fixed window with attention: every position asks all earlier
positions "how much do you matter to me?", gets a learned mix of their
contents, and predicts from that mix. A causal mask cuts every look
forward, so a prediction never sees its own answer.

## The pieces

- `murrai/domain/attention.py`: `CausalSelfAttention` with `heads`
  heads (default 1). The match score between positions is a dot product,
  scaled down by the square root of the head's width so long rows don't
  saturate the softmax. The mask sets future matches to −∞ before the
  softmax, making them weigh zero.
- `murrai/domain/block.py`: `TransformerBlock` — attention, then a
  small per-position network (one hidden layer, four times wider than
  the model, GELU (Gaussian error linear unit) instead of tanh), each
  wrapped in layer-norm-first
  ("pre-norm") and a residual: the block's output is added to its
  input, so a block can choose to change almost nothing.
- `murrai/domain/transformer.py`: `TransformerLM` (`kind` "char-gpt")
  — a character embedding plus a learned table of position numbers,
  then N blocks, a final layer norm, and a linear head to character
  odds. Same surface as `MlpLM`: logits, probs, nll, perplexity,
  generate, state/from_state.
- `murrai/application/runs.py`: `train_transformer_lm` — the mlp
  trainer with two swaps: Adam instead of SGD (attention trains badly
  with plain gradient descent), and each batch is 256 chunks of 64
  consecutive characters scored at every position at once, instead of
  256 separate windows. The plateau rate schedule carries over
  untouched.
- `murrai/composition/chat.py`: the "char gpt" stage, saving
  checkpoints as `char-gpt`.

## Heads

The model's `dimension` numbers are cut into `heads` equal slices. Head
i reads only slice i of the query and the key, builds its own table of
scores with its own causal mask and softmax, and blends only slice i of
the value. The blends from all heads are glued back into `dimension`
numbers and pass through the `out` layer. The scale becomes one over the
square root of the slice width (`dimension / heads`), not of `dimension`.
The weights are the same four square layers as before, so the parameter
count does not change with the head count. `dimension` must divide
evenly by `heads`, or construction raises `ValueError`.

The point of heads is that each one can learn a different pattern of
which positions to read. `attention.shares` holds each head's table, with
shape (batch, heads, time, time).

Checkpoints store `heads`. A checkpoint saved before heads existed has no
such entry and loads as 1 head.

## How scores are computed (the chunking convention)

The transformer's window is 64 characters, and held-out text is scored
in non-overlapping chunks of 64: one forward pass per chunk, every
position of the chunk scored in that pass. Consequence: a character
early in a chunk is predicted with almost no context (the chunk just
started), while the mlp and the bigram score every character against
its full history. This slightly inflates the transformer's perplexity.
That is the honest direction for a comparison: if the transformer wins
while carrying the handicap, it wins for real.

## Evidence

Corpus: the first 300,000 characters of `data/corpus/tinystories.txt`,
last 10% held out, seed 0 for training, seed 17 for the samples.

| model | held-out perplexity | train loss |
|---|---|---|
| counted bigram | 10.034 | - |
| windowed mlp | 3.459 | 4.225 -> 1.124 |
| char gpt | **3.029** | 4.406 -> 0.915 |

The char gpt beats the windowed mlp while scoring under a handicap, on
the same split. Configs: char gpt is 4 blocks, 64 dims, window 64
(~280k parameters, 68-char vocabulary), 2,000 steps, batch 256, Adam
rate 1e-3 with the plateau shrink, about 70 ms per step on CPU, under
3 minutes. The mlp is the phase-2 config (window 8, 32 dims, 256
hidden, 10,000 steps, batch 256, rate 0.5). Each model saw the same
number of characters; the transformer saw fewer passes over them and
still won.

Three-line samples, 200 characters each, same seed (17):

```
counted bigram:
0Ef. sk an s lbed shitr. Thagoven. ane t han's hapirink anacathe ily aithentond n, g oft Herd o fte.
Ond Ito al
Ticeore thize ut he l He totime sopilou ked. a hey w the igvepa. tedan ct brsay. oule b

windowed mlp:
"EOnk, for halbed the boy named Timmy. Timmy's mom is.
Max and have in the conden, swo, noxad Cister she was a boy named Timmy. Timmy loud ott. I day, Lily and Max werectight askvery big breable. He w

char gpt:
ing. Timmy said: "What's play!" her friends Max anymorry. Max was sad. When it was good, but he fish she was all happy to hide the man in too. One day, Lily saw Lily was stick. She anymore says, "Don'
```

The gpt's text carries the corpus's shape — names, quoted dialogue,
story openings — further than the mlp does, and both are far past the
bigram's letter soup.

### Heads

Same split and seed as the char gpt row above, with only the head count
changed. Everything below is one run per cell. We did not measure how much
two seeds differ at one setting, so gaps of a few hundredths may be noise.

**Small model** (4 blocks, 64 dimensions, window 64, 2,000 steps, batch
256, about 220,000 parameters). Held-out perplexity, lower is better:

| text | 1 head | 4 heads | 8 heads |
|---|---|---|---|
| first 300,000 characters of TinyStories | **3.029** | 3.053 | 3.094 |
| all of TinyStories (2M characters) | **3.236** | 3.282 | 3.316 |
| every file in `data/corpus` (26.5M characters) | 4.221 | **4.216** | 4.311 |

More heads did not help here. Four heads tied or lost on all three
texts, and 8 heads lost on all three. With 64 numbers to share out, 4
heads get 16 each and 8 heads get 8 each.

**Larger model** (6 blocks, 384 dimensions, window 256, batch 64, about
10.9 million parameters), trained on every file in `data/corpus`, on the
Apple GPU. The first two rows ran at the same time on one GPU:

| heads | slice width | steps | held-out perplexity | train loss, last 100 steps |
|---|---|---|---|---|
| 1 | 384 | 2,000 | 3.216 | 1.190 |
| 6 | 64 | 2,000 | 3.155 | 1.155 |
| 6 | 64 | 4,000 | **2.951** | 1.058 |

At this size 6 heads beat 1 head by 0.061 perplexity (1.9%) with exactly
the same number of parameters. The 6-head loss curve sits below the 1-head
curve at every 100-step mark from step 400 onward, by 0.03 to 0.04, so
the gap is steady, not a lucky final number. Our reading is that a head
needs room: 64 numbers per head work, 16 or 8 do not. That is untested
beyond these two sizes.

Doubling the steps lowered perplexity by a further 0.204. The loss
stopped moving at about 1.055 from step 3,100 on. The plateau schedule
shrinks the learning rate each time progress stalls, so the flat stretch
may come from the schedule and not from the model's limit. That is
untested.

The held-out text is the end of the folder, which sorts to *Walden* and
*Wealth of Nations*. The model never saw those books.

Run any row with `uv run python scripts/train_heads_demo.py <directory of
.txt files> --heads N`. It also takes `--steps`, `--depth`, `--dimension`,
`--window`, `--batch`, `--chars` (use only the first N characters),
`--device mps`, and `--save NAME` to keep the model in the checkpoint
store. The larger model's flags are `--depth 6 --dimension 384 --window
256 --batch 64 --device mps`.

A 200-character sample from the 6-head, 4,000-step model, seed 17:

```
's of living council so."


28. Of the conclusions in some of this wrong and buting their floates
of reason, does not enough give the justice of a corn nor
the river, for both of political laws, turn
```

Words are mostly real, short stretches read as English, and the meaning
does not hold across a sentence.

## What it unlocks

The window is gone: context length is now a parameter (`max_time`), and
attention decides per position what to read. Phase 4 builds our own
tokenizer, which cuts text into frequent chunks instead of single
characters. Phase 5 scales the model to 10 to 50 million parameters and
scores it against GPT-2-small (ADR-0004). The 10.9 million parameter rows
above are a first look at that size.

## Resources

- Intro: Wikipedia, [Transformer (deep
  learning)](https://en.wikipedia.org/wiki/Transformer_(deep_learning)).
- Middle: 3Blue1Brown, [Attention in transformers,
  step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc): the visual
  pass over queries, keys, and values. Karpathy, [Let's build GPT: from
  scratch, in code, spelled
  out](https://www.youtube.com/watch?v=kCc8FmEb1nY): the same pieces
  built in code.
- Advanced: Vaswani et al., [*Attention Is All You
  Need*](https://arxiv.org/abs/1706.03762) (2017): the original paper.
  Ba, Kiros, Hinton, [*Layer
  Normalization*](https://arxiv.org/abs/1607.06450) (2016): the
  normalizer the blocks use.
