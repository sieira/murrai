# Study: the model's anatomy

What the model is inside, what runs when one character goes in, and what
is and is not random about it.

## What the model is right now

One table of floats, shape (118, 117): one row per previous character
(117 characters plus one start marker), one column per possible next
character. 13,806 numbers in total. No layers, no hidden units, nothing
else. The extra dimensions of later models (activations shaped by batch
and position, attention tables shaped by heads and positions) do not
exist yet.

## Input "a": the operations

1. The vocabulary turns the character into its id, a row number.
2. Row pick: take that row's 117 scores. In the counted model this is
   indexing; in the neural telling, one-hot times the weight matrix
   multiplies out to exactly the same row, which is why the multiply is
   skipped.
3. exp each score, then divide each by the total: 117 probabilities
   summing to one. The counted model has this pre-baked into the row.
4. Writing text rolls a weighted die over those probabilities. Scoring
   text never rolls it: it reads the probability the real next
   character received.

Measured on the Aesop corpus, after "a": "n" 0.252 ("and"), "t" 0.118
("at"), "r" 0.089, "s" 0.082, space 0.062.

## Why e^loss is the number of choices

One fact underlies the metric: logarithms convert multiplication into
addition, and exponentials convert it back.

    log(x * y) = log x + log y
    e^a * e^b = e^(a+b)

Each +1 in an exponent is one multiplication by e outside; the log
answers "how many multiplications by e reach this number". They are the
two directions of one bridge between additive and multiplicative land.

A text's probability is a product: p(char 1) times p(char 2), and so on.
Long products collapse toward zero and compare badly; taking the log
turns the product into a sum of per-character costs, which is what
`nll` (negative log-likelihood) computes. Training and scoring live in
this additive land.

e^loss translates back to multiplicative land. With an average cost of
2.48 per character, each character multiplies the text's probability
down by e^2.48, about 12. Perplexity is that shrink factor per
character.

The "equally likely choices" reading follows by matching a uniform die:
an honest die with N faces multiplies probability by 1/N each step,
which costs log N per step. Matching a model with average cost L to
such a die gives log N = L, so N = e^L: the size of the uniform die
exactly as surprised as the model. The counted bigram's 2.48 is a
twelve-faced die; the blind model's 4.76 is the full 117.

## Determinism, in three layers

1. The operations are deterministic. The same character in produces the
   same probability row on every run; exp and divide roll no dice.
2. Generation is random by design. The model's output is a distribution
   over next characters, not one character, and language is a
   distribution, not a function: many continuations of the same text
   are valid. Removing the die and always taking the most likely
   character collapses into loops: measured output was
   "Lind the the the the the the the the the". The die preserves the
   corpus's variety. Temperature and top-k, when they arrive, change
   the die, never the table.
3. The randomness itself is replayable. The die is a pseudo-random
   generator seeded with a number: seed 17 twice produced identical
   text; seed 99 differed. Same for training's minibatch draws. Same
   seed, same run.

The non-determinism seen across runs is the seed changing, never the
table changing.

## Resources

- Intro: Wikipedia,
  [softmax function](https://en.wikipedia.org/wiki/Softmax_function) and
  [perplexity](https://en.wikipedia.org/wiki/Perplexity).
- Middle: Jurafsky, Martin, [N-gram Language
  Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (chapter 3 of
  *Speech and Language Processing*, free online): the textbook's
  treatment of perplexity, for word models instead of characters.
- Advanced: Shannon, [*A Mathematical Theory of
  Communication*](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
  (1948), where measuring English's predictability began.
