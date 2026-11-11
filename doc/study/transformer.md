# Study: the transformer

The transformer is the architecture that stacks attention into a deep
network. Characters become vectors that carry positions, flow through a
run of identical blocks, and end at a projection to next-character
odds. Each block lets positions exchange information through attention,
then processes every position alone through a small per-position
network. This note covers the architecture around the mechanism; the
mechanism itself and the evidence for the whole assembly live in
[attention.md](attention.md).

## Why one attention layer is not enough

One attention layer gives every position one weighted look at its
earlier positions. That answers what surrounds each position, not what
to conclude from it. The gathered information arrives raw, and a single
layer has no step left to process it before the prediction. Reading
text takes alternating moves: gather from the earlier positions,
process what was gathered, gather again. Stacking identical blocks
gives the network those alternating rounds, and depth, the number of
blocks, sets how many.

## The block

The unit of depth is the block (`murrai/domain/block.py`). A block
performs two operations on the stream of vectors and wraps each one the
same way:

1. Attention. Positions read from earlier positions, exactly as
   attention.md describes.
2. A per-position mlp (multi-layer perceptron). Every position, alone
   and independent of the others, passes its vector through one hidden
   layer, four times wider than the model's dimension, with GELU
   (gaussian error linear unit) nonlinearity.

The wrapping works the same for both operations. Before each one, layer
normalization rescales the stream (each vector is shifted and scaled so
its entries average 0 and vary by 1). Normalizing before the operation
rather than after is the "pre-norm" arrangement, and it is what lets
deep stacks train without learning-rate tricks. After each operation,
the result is added back to the operation's input instead of replacing
it. That addition is the residual. The block's input reaches its output
unchanged unless the block chooses to change it, so every block only
ever contributes a correction, and each block carries two of these
add-back paths, one around the attention and one around the mlp. The
same additions carry the learning signal backward, because a signal can
reach any depth without passing through every transformation. That is
why a stack of 4 blocks trains as ours does, and why stacks of 40 do
too.

## The bookends

At the bottom, two lookups build each position's starting vector: the
character's row in a learned embedding table (the table has one extra
row for the start-of-text marker), plus that position's row in a second
learned table with one row per slot in the 64-character window. The
second table exists because attention is order-blind. It scores matches
between vectors but never sees which slot each position sits in, so the
model must be told the order explicitly. At the top, after the last
block, one more layer normalization rescales the stream a final time,
and a linear head projects every position's vector to odds over the
vocabulary. Training scores every position of the window at once, so
the loss comes from every position, not only the last one.

## Why this shape trains

- Every operation is a matrix product or a table lookup. There is no
  recurrence, so all positions of a window are computed in one pass.
- Pre-norm keeps the stream's scale stable as the stack grows.

## What we built against the paper

Vaswani et al. (2017) introduced the architecture with multi-head
attention. Several attention patterns run in parallel inside each block
and their outputs are concatenated. murrAI's char gpt keeps one head
per block, 4 blocks, 64 dims, and a 64-character window, about 280k
parameters. Multi-head is named here and deferred. It gets its own gate
session before it exists. The rest of the shape, blocks, pre-norm,
residuals, learned positions, is the paper's.

## Resources

- Intro: Wikipedia, [Transformer (deep
  learning)](https://en.wikipedia.org/wiki/Transformer_(deep_learning)):
  the architecture overview.
- Middle: 3Blue1Brown, [Attention in transformers,
  step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc): the
  visual pass over queries, keys, and values. Karpathy, [Let's build
  GPT: from scratch, in code, spelled
  out](https://www.youtube.com/watch?v=kCc8FmEb1nY): this exact stack,
  built in code.
- Advanced: Vaswani et al., [*Attention Is All You
  Need*](https://arxiv.org/abs/1706.03762) (2017): the original paper.
  Ba, Kiros, Hinton, [*Layer
  Normalization*](https://arxiv.org/abs/1607.06450) (2016): the
  normalizer the blocks use.
