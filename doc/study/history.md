# Study: how next-token prediction became reasoning

The history behind the recipe: where the ideas came from, what each step
actually claimed, and why nobody saw the destination coming.

## The short answer

Nobody knew. Nobody set out to build reasoning. For seventy years, every
generation's goal sat one step lower: compress telegrams, recognize speech,
translate, finish sentences. Each step paid its own rent, which is why
thousands of people kept working on formulas whose destination they could
not see. Reasoning arrived in 2020 as a surprise inside a next-word
predictor, found by people poking at it, not designed by them.

## What the machine does, in one sentence

A natural first description: "adjusting a bunch of parameters so when
they do a bunch of multiplications it ends up mostly outputting what is
more probable to follow." Two sharpenings make it exact:

- The model never outputs one answer. It assigns a probability to every
  possible next character. Picking one is sampling, a separate step.
- Training does not chase "what is probable" in the abstract. It pushes
  the model to give high probability to what actually followed in the
  corpus. The corpus is the teacher; the loss measures the miss; the
  gradient tells every parameter which way to nudge; thousands of rounds.

## The timeline, and what each step actually claimed

1. Shannon, 1948. He chased telegrams, not thought. To measure how
   compressible English is he built the counted bigram (and its bigger
   siblings, which read two and three previous characters), and had humans
   guess the next letter: real English carries roughly one bit of
   information per character, about three quarters predictable from
   context. The claim was modest: text has statistical structure you can
   measure and exploit. Nothing about meaning.
2. The 1950s to 1980s. The dominant bet for language was hand-written
   logic rules. That fork consumed a generation and mostly failed at
   language. The learning fork was declared near-dead in 1969 and survived
   in niches: digits, speech.
3. Backpropagation, 1986. Networks learn internal representations by
   nudging weights against their gradient. Claim: learnable
   representations, demonstrated on digits and speech. This is our
   micrograd engine.
4. Bengio, 2003. Predict the next word with a neural net instead of a
   tally. Claim: it beats counting, because learned vectors let similar
   words share strength ("cat" and "cats" stop needing separate counts).
5. word2vec, 2013. King minus man plus woman lands next to queen. Nobody
   programmed meaning; the geometry learned from prediction carried it.
   This is when the deeper bet became credible to serious people.
6. The transformer, 2017, and GPT-2, 2019. Same recipe, scaled. GPT-2 was
   considered so uncannily good at continuing text that its release was
   staged as dangerous. By today's standard it is a toy; expectations
   moved that fast.
7. GPT-3, 2020. The reasoning-adjacent abilities (arithmetic, translation,
   following a pattern shown twice) were discovered by users, not built by
   the designers. The builders were surprised too. "Emergence" is the
   honest name for: it happened, and we lack the theory.
8. ChatGPT, 2022. The reasoning was already in the base model. The
   assistant shape was grafted on by tuning on human dialogues.

## The parallel track: control rooms, not chatbots

The timeline above is the language line of work. It was not the only
one. Through all those years, the same math ran a second career in
engineering, where nobody expected thinking and nobody was disappointed.

- Zadeh, 1965: fuzzy sets, degrees of truth instead of yes/no membership.
  Rules written by hand ("IF temperature IS hot THEN fan IS fast"),
  executed with arithmetic. Branded "approximate reasoning".
- Fuzzy control shipped in industry from 1987: the Sendai subway's
  braking, cement kilns, camera autofocus, washing machines, anti-lock
  brakes.
- Neural nets as function approximators: the universal approximation
  theorem (1989-1991) proved one hidden layer can represent any
  continuous function, and backprop trained useful ones for system
  identification and control. ANFIS (adaptive neuro-fuzzy inference
  system; 1993) fused the two: a neural net tuning the shapes of
  hand-written fuzzy rules.
- By the 2000s this track had a name, "soft computing", and its own
  courses. In Spanish-speaking universities, "control neuronal y
  borroso" was a standard elective in industrial and telecommunications
  engineering around 2007, because industry used it: process control,
  fault detection, sensor fusion, pattern recognition.

By 2007 the concepts were standard engineering curriculum, and they were
artificial intelligence material in the engineering sense: known, taught,
and applied to real systems. What they were not yet was a path to
language. Their problems were low-dimensional and smooth (a
kiln's temperature curve, a motor's dynamics), their nets were small,
their rules were hand-authored. Language is the opposite profile:
enormous dimension, discrete symbols, and the "function" is the whole
regularity of human text. Three ingredients were missing: web-scale
text, GPUs (graphics processing units), and the bet tested at scale.
They arrived after Hinton's 2006 result on training deep nets, and the
first visible win was vision, not language: AlexNet, 2012, trained on
two consumer graphics cards.

Two pieces of that curriculum reappeared inside transformers. Hopfield
networks (1982), which store memories as attractors a partial pattern
settles into, returned in 2020 when researchers showed that attention is
a modern Hopfield update. Kohonen's self-organizing maps (1982), which
place similar inputs near each other on a grid, are ancestors of
embedding thinking.

The irony worth keeping: Zadeh branded fuzzy logic "computing with
words". The 2007 course taught reasoning-shaped machines whose rules a
human wrote. Language models vindicated the goal by abandoning the
mechanism: they compute with words, with no authored rules anywhere.

## What reasoning is made of

Handling imprecise rules sounds like a partial definition of reasoning.
It is one, and Zadeh agreed: "approximate reasoning" was his own
branding. Making the "partial" precise locates the whole history:

1. A calculus for the unclear. Two unclarities, often confused:
   vagueness (the predicate has soft edges: is 38.2 degrees "hot"?
   partially) and uncertainty (the fact is crisp but unknown: will the
   patient fever tomorrow? that is probability). Fuzzy logic attacked
   vagueness; probability attacked uncertainty. Solved by the 1980s, and
   only the form of reasoning.
2. Content: what things are, what causes what, which predicates exist.
   No calculus supplies it. Expert systems spent the 1980s hand-writing
   rules and drowned in the authoring.
3. Composition: chains of steps, recombined in ways no author
   anticipated. A flat rule base cannot climb, however well it handles
   "almost".
4. Acquisition: getting the first three from experience instead of from
   a programmer.

Fuzzy logic attacked one slice of the first component and had no story
for the rest; that is why "control neuronal y borroso" was engineering
and not reasoning: same calculus, empty container.

Language models got components two, three, and four from prediction
pressure instead of authorship, and the first component fell out as a
byproduct: a model knows "almost done" versus "done" because the
corpus's statistics forced the distinction into its weights, not because
anyone wrote a membership function. The calculus Zadeh hand-crafted
turned out to be learnable as a side effect of learning content at
scale. That inversion is the deepest turn in this history.

The mapping onto this repo: in `NeuralBigram`, the calculus is the fixed
code (softmax, negative log-likelihood, the gradient machinery) and the
content is the learned weight table. The history since 2017 is a change
in that ratio: the calculus barely moved while the content grew from a
117-character table to hundreds of billions of parameters. The recipe
stayed the size of a page; the numbers became the size of the world.

## Where the probability half came from

Language is fundamentally uncertain: many continuations are valid, so any
machine that handles language must handle uncertainty, and probability is
the only consistent math of uncertainty (Jaynes framed it as logic
extended to cover unknowns). That half was settled long before anyone
thought of language models. The language half came from Shannon. Nobody
combined them into a bet about reasoning until decades later; the
combination was assembled step by step, for local practical reasons.

## The bet under everything

To predict text well, you must capture the regularities of the world that
produced the text. Language compresses human experience; relentless
prediction pressure forces a machine to build internal grammar, facts,
causality, psychology, whatever lowers surprise. The formal ancestor is
Solomonoff, 1960s: the ideal predictor of a sequence is its shortest
explanation, and a shortest explanation is what we mean by understanding.
The slogan is "compression is intelligence". Scaling laws are the measured
version of the bet: loss falls predictably as size and data grow.

## The counter-bet

Chomsky, from 1957: language is rule-based recursion; statistics cannot
reach syntax, let alone meaning. That position was the mainstream for
sixty years, not the fringe. The outcome settled the argument empirically
and recently, by results rather than by anyone's foresight. Both bets
were philosophical positions carried by people who kept working anyway;
the evidence arrived last.

## What is still unknown

There is no accepted theory of why it works. We have the curves (loss
falls predictably with scale), the surprises (abilities appear at sizes
nobody predicted), and a live disagreement about what the weights contain:
recited text, or a model of the world. The metaphysics is current work,
not settled history.

## The mirror in this repo

The counted bigram is Shannon's step. The neural bigram trained through
micrograd is 1986 meeting 2003. The remaining phases retrace the sequence
on purpose: the MLP (multi-layer perceptron; Bengio's step), attention
(2017), then scale (2020).
The phase plan in SPEC.md is this history, replayed at desk size.

## Resources

- Intro: Wikipedia, [history of natural language
  processing](https://en.wikipedia.org/wiki/History_of_natural_language_processing).
- Middle: Karpathy, ["[1hr Talk] Intro to Large Language
  Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g): a general-audience
  one-hour introduction, his own description. And [*Neural Networks: Zero
  to Hero*](https://github.com/karpathy/nn-zero-to-hero): the
  build-it-yourself version of the same sequence, already the reference
  for the other study notes.
- Advanced: Shannon, [*A Mathematical Theory of
  Communication*](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
  (1948): the counted bigram and the predictability measurement, in the
  original. Jaynes, [*Probability Theory: The Logic of
  Science*](https://bayes.wustl.edu/etj/prob/book.pdf), chapter 1:
  probability as extended logic, the frame under all of it. Zadeh,
  [*Fuzzy sets*](https://doi.org/10.1016/S0019-9958(65)90241-X) (1965):
  the "approximate reasoning" branding. Bengio, Ducharme, Vincent, Jauvin,
  [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  (2003): the paper that started predicting language with a neural net.
  Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
  (2017): the transformer, the paper behind phase 3.
