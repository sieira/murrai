# Study: bigram language models

The theory behind the counted bigram and the neural bridge to it. Plain
language stays here; the code and the glossary use the same words.

## The job

A language model does one thing: look at the text so far and assign a
probability to every possible next character. Every model in this project
answers the same question with a bigger memory.

## The bigram answer

"Bigram" = two in a row: the model's whole memory is the previous
character. Building it from a corpus:

1. Walk the corpus and tally each (previous, next) pair of characters.
2. Turn each row into probabilities by dividing by the row total.
3. Add 1 to every tally first (smoothing), so no row is ever all zeros
   and no next-character is impossible by construction.

That table is the model. Generating samples from it: start from a start
marker, read the row, roll a weighted die (sampling), write the character,
repeat. Evaluation never rolls the die: negative log-likelihood asks
"how unlikely was the truth" over a held-out slice, and perplexity is that
converted to equally-likely choices per character.

What a bigram can never do, at any corpus size: remember more than one
character back. It cannot know "q" is near-surely followed by "u" a step
later, and nothing later than one step.

## Why the table is secretly a neural net

A one-hot vector (N zeros, one 1 in the previous character's slot)
multiplied by a matrix picks out one row of that matrix. Softmax of that
row gives the same probability row the counted table gives. So the table
is the smallest possible neural network with the right weights, and the
weights can be *learned* instead of tallied: start from any matrix, define
the loss as negative log-likelihood, and repeatedly nudge every number a
small step (the learning rate) against its gradient (how the loss would
move if that number moved). Converged, the learned matrix reproduces the
counted table.

Counting cannot be extended to attention; gradient descent can. That is
the reason the project builds an autograd engine: the same nudge loop
carries from bigram to transformer unchanged.

## In this repo

- `murrai/domain/` holds the tallying, probabilities, sampling, and
  perplexity math (pure, offline).
- `murrai/adapters/` reads corpora from files.
- `murrai/application/` runs a training pass: corpus in, bigram built,
  held-out perplexity out.

The training corpus for phases 1-5 is TinyStories (decision recorded in
this file because the choice follows SPEC phase 5 and ADR-0004; the full
naming and licensing ADRs stay at Phase 5). Until the TinyStories
download adapter exists, `make corpus` fetches a public-domain slice —
Aesop's Fables, Townsend translation (Project Gutenberg #21) — into
`data/corpus/`. Nothing dataset-sized is committed; no training data
lives in the repo.

## Resources

- Karpathy, *Neural Networks: Zero to Hero*, lesson "makemore part 1":
  a char-level bigram built by counting and rebuilt as a neural net, the
  exact bridge described above.
- minbpe and nanoGPT come later (phases 4 and 3 respectively).
