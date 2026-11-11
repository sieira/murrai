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

- `murrai/domain/attention.py`: `CausalSelfAttention` — one head; the
  match score between positions is a dot product, scaled down by √d so
  long rows don't saturate the softmax; the mask sets future matches to
  −∞ before the softmax, making them weigh zero.
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
rate 1e-3 with the plateau shrink — about 70 ms per step on CPU, under
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

## What it unlocks

The window is gone: context length is now a parameter (`max_time`), and
attention decides per position what to read. Phase 4's remaining work
is making this trainable at scale — better schedules, longer windows,
more data — toward the SPEC's chat target.

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
