# Study: universal approximation

The theory behind "what are we searching inside": the family of functions
a network can represent, and what its universality does and does not
promise.

## The definition

Take any continuous function (no jumps) on a bounded box of inputs, and
any tolerance, say 0.001. The theorem: there exists a choice of weights
for a network with at least one hidden layer and a non-polynomial
activation, such that the network's output stays within that tolerance
of the target everywhere in the box. The family of functions reachable
by turning the weights is dense in the continuous functions: no
reasonable target on a finite box escapes it.

One line, formal: the family { x -> sum_i c_i * sigma(w_i . x + b_i) }
is dense in C(K), the continuous functions on a compact set K. Cybenko
proved it for sigmoid activations (1989); Hornik widened the conditions
(1991); Leshno, Lin, Pinkus, Schocken found the exact border: any
non-polynomial activation works, and no polynomial one does (1993).
ReLU (rectified linear unit), which is not even smooth, needed separate
work (2017).

## The Fourier parallel

Fourier: any reasonable function on an interval is a sum of sines with
free coefficients. The sines are frozen; the knobs are the coefficients.
A network does the same with soft steps instead of sines: each hidden
unit contributes one sigmoid(w . x + b), a curve whose position and
steepness are knobs, and the output layer adds the contributions. Enough
steps and ramps trace any continuous landscape. Two differences: the
network's basis functions themselves carry knobs (Fourier's sines do
not), and Fourier's theorem is constructive (computing the coefficients
is straightforward) while approximation is existence only (it says
nothing about finding the weights).

## What the theorem does not say

- Which weights. It is existence, not construction.
- How many units. The proofs tolerate astronomical widths; no usable
  count comes out of them.
- Whether training finds them. Nothing connects the theorem to gradient
  descent.

Representation is not learnability. The theorem was on curriculum by the
early 1990s; the language model arrived thirty years later, when the
empirical answer landed: gradient descent, given web-scale text and GPU
(graphics processing unit) compute, does find usable weights. The proven
half was the easy half. This is the same gap as the bet in the history
notes: the claim under everything is learnability, and it is carried by
evidence, not by theorem.

## Why the nonlinearity cannot go

Without an activation between layers, a stack of matrix multiplies is
itself one matrix multiply, and a linear model cannot even represent the
square of its input. The activation is what lets each layer bend the
space.

Two footnotes. Depth: two layers suffice eventually, but some targets
need exponentially more units in one hidden layer than in a few stacked
ones, which is why practice grows deep, not just wide. Discreteness: on
token inputs the theorem turns near-trivial, since a large enough table
can memorize a finite set of cases; its real content is the continuous
case, and its real lesson stays representation-versus-learning.

## Where this repo sits

The neural bigram has zero hidden layers: one-hot times the weight
matrix only picks a row, so its function family is exactly the set of
one-step memories. That is the formal version of "it cannot look back".
The theorem starts to apply at phase 2: the hidden layer of the MLP
(multi-layer perceptron) builds learned basis functions over the context
window. The transformer stacks the same idea.

## The three universals

Three different promises hide behind the word "universal":

- universal function family: this theorem, about representation.
- universal computer: Turing (1936), one machine that runs any program.
- universal predictor: Solomonoff (1960s), one distribution that
  compresses any sequence; the bet under everything in the history
  notes.

Language models needed all three at once: a universal family, trained by
a learnable procedure, under a prediction objective.

## Resources

- Intro: Wikipedia, [universal approximation
  theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).
- Middle: Nielsen, [chapter 4](http://neuralnetworksanddeeplearning.com/chap4.html)
  of *Neural Networks and Deep Learning*: a visual proof sketch of the
  theorem, free online.
- Advanced: Cybenko, [*Approximation by superpositions of a sigmoidal
  function*](https://doi.org/10.1007/BF02551274) (1989): the original
  statement. Hornik, [*Approximation capabilities of multilayer
  feedforward networks*](https://doi.org/10.1016/0893-6080(91)90009-T)
  (1991): the widening. Leshno, Lin, Pinkus, Schocken, [*Multilayer
  feedforward networks with a nonpolynomial activation function can
  approximate any function*](https://doi.org/10.1016/S0893-6080(05)80131-5)
  (1993): the non-polynomial border.
