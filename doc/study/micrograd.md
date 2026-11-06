# Study: the autograd engine (micrograd)

Notes from the author-understanding session before the engine was built.
Plain language throughout; the code and the glossary use the same words.

## The job

A calculator answers one question: given numbers, what comes out. The
engine answers two: what comes out, and how sharply each input number
moved the answer (the gradient). All of training rides on the second
answer, because the training loop's rule is simple: nudge every knob a
small step against its gradient, thousands of times, and watch the loss
fall.

## The Value mechanic

A `Value` wraps one float and remembers three things a float does not:
its children, the operation that produced it, and a gradient slot.
Arithmetic on Values does not compute a result only; it files the recipe.
`backward()` then walks the recipe tree backwards (topological order:
parents before children) and every node does one small local job: take
the gradient that arrives from above, multiply by its own local slope,
pass it on. In `c = a * b` the local slope of `a` is `b`, and so on;
the chain rule is this pass-along in mathematical clothing.

Two rules the walk forces:

- A number reused on several paths adds its gradients from every path
  (accumulation, `+=`).
- Because slots accumulate, they must reset to zero before each round of
  nudging, or stale gradients leak between rounds.

## Two honest methods, cross-checking

- Brute: nudge one number by a hair, rebuild the loss, divide the change
  by the nudge (finite differences). Never wrong, brutally slow: one
  rebuild per number.
- Walk: `backward()`. One sweep, exact by construction.
The repo's verifier (`finite_difference_grads`) makes the two agree on a
seeded graph, and a test makes the engine agree with torch at 64-bit
precision. Trust in the walk is bought from the brute method, not claimed.

## What was built

- `murrai/domain/autograd.py`: the `Value` class (ops `+ - * / ** tanh
  exp log`, right-hand variants, `backward()`, `zero_grad()`) and the
  finite-difference verifier.
- `murrai/domain/neural_bigram.py`: a bigram whose table is not counted
  but trained: one row of logits per previous character, softmax over the
  row, loss the negative log-likelihood of the real next characters.
- `murrai/application/runs.py`: `train_neural_bigram`, the nudge loop:
  sample a minibatch of pairs from the corpus, run the loss through the
  engine, apply the learning rate against every gradient, reset, repeat.

## Evidence (Aesop train slice, 46,885 chars, 117-char vocab)

Loss curve across 2,000 steps (seed 0, batch 64, rate 3.0 decaying to
0.1): 4.75 (blind uniform) -> 2.71, against the counted table's exact
2.48. The learned table sits on the counted table: average probability
row distance 0.20, worst row 0.36, over rows with 50+ uses. Gap between
2.71 and 2.48: the last stretch of convergence plus minibatch sampling
noise, both costs of learning, not errors of the engine (the engine's
own gradients are exact up to floating point, as the verifier shows).

Samples at seed 17 look alike for both models - crude but structured
character salad, with enough English skeleton (word shapes, spaces,
punctuation) to show the distribution was truly learned:

```
counted: 0"uke ms and Oome moty wl.Hecrigesesendiferoure pr,[,...
trained: moaMamae4(XglkIpo[" toPoqus-Dous Hahh-8ggRi!W-yU-Wgr...
```

## What it unlocks

Phase 2 replaces counted tables with trained parameters, using torch's
C++ version of the same walk (`loss.backward()`). The loop, the loss,
the learning rate, and the reset stay identical; only the speed of the
engine changes.
