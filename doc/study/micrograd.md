# Study: the autograd engine (micrograd)

The theory behind the autograd engine: what it computes and why it can
be trusted. Plain language throughout; the code and the glossary use the
same words.

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

## The update rule

One formula moves every weight, every step:

    w <- w - rate * gradient(w)

In code, `update()` in `neural_bigram.py`: `weight.data -= learning_rate
* weight.grad`.

- gradient(w): how much the loss moves per unit increase of w; the walk
  computes it, summed over every path through w in the minibatch, and
  `batch_loss` has already divided by the batch size.
- rate: one shared boldness knob, the same for every weight.
- minus: the gradient points uphill by definition, so descent steps the
  other way. First-order justification: L(w + d) is approximately
  L + d * gradient(w); choosing d = -rate * gradient(w) predicts a
  change of -rate * gradient(w)^2, never positive, whatever the
  weight's sign. One rule, every weight.

The linear prediction holds only near the current point, so the rate
must be small. Measured on one batch of 8 pairs: rate 0.5 lowered a
fresh batch from 4.76 to 4.75, rate 50 to 4.48; rate 500 left the
trained pairs at 0.17 but pushed the fresh ones to 11.47, worse than
the blind 4.76. A giant step memorizes the sampled pairs and wrecks
everything else. The run's decay from 3.0 to 0.1 is this table in
motion, and the same measurement is the held-out lesson in miniature:
the trained batch flatters any rate; only fresh pairs tell the truth.

## What it unlocks

Phase 2 replaces counted tables with trained parameters, using torch's
C++ version of the same walk (`loss.backward()`). The loop, the loss,
the learning rate, and the reset stay identical; only the speed of the
engine changes.

## Resources

- Intro: Wikipedia,
  [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
- Middle: 3Blue1Brown, ["Backpropagation,
  intuitively"](https://www.youtube.com/watch?v=Ilg3gGewQ5U) and
  ["Backpropagation calculus"](https://www.youtube.com/watch?v=tIeHLnjs5U8)
  (chapters 3 and 4 of their deep learning series). Karpathy, ["The
  spelled-out intro to neural networks and backpropagation: building
  micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0), the lesson
  this engine mirrors. It asks for Python and "a vague recollection of
  calculus from high school", his words.
- Advanced: Rumelhart, Hinton, Williams, [*Learning representations by
  back-propagating errors*](https://www.nature.com/articles/323533a0)
  (1986), the demonstration that the walk trains layers.
