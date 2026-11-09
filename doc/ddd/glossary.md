# Glossary

The ubiquitous language. Identifiers in code, tests, and docs use these
terms. Entries are added at decision time, one plain line each. This file
binds (AGENTS.md header).

## Numbers and structures

- tensor: a multi-dimensional array of numbers; everything in murrAI is one.
- parameter: one learned number inside the model; murrAI targets 10-50M.
- corpus: the training text; TinyStories is the corpus for phases 1-5.
- token: one integer the model reads or predicts; text becomes tokens.
- one-hot: a vector of zeros with a single 1 marking one token id.
- bigram: a model whose whole memory is the previous token; our baseline.
- neural bigram: the same memory as a bigram, but its table is trained
  by gradient descent through the autograd engine instead of tallied.
- tokenizer: the pair of functions turning text into tokens and back.
- BPE: byte pair encoding; the algorithm our tokenizer learns as merges of
  frequent pairs.
- embedding: a learned vector standing for one token; a lookup row
  fetched by index, describing the token rather than predicting.
- context window: how many previous tokens the model can see at once.
- concatenate: lay vectors side by side into one long vector; how a
  context window enters the network, each position keeping its own
  weights.
- MLP: multi-layer perceptron; a network of lookup, linear layers, and
  an activation between them.
- logits: raw scores, one per vocabulary entry, before normalization.
- softmax: turns logits into probabilities that sum to one.
- attention head: one learned pattern of "which tokens look at which".
- batch: a group of sequences trained on at once.

## Training

- loss: one number measuring how wrong the predictions are; training
  lowers it.
- negative log-likelihood: the loss for a language model; how unlikely
  the real next token was under the model's distribution.
- cross-entropy: the torch name for negative log-likelihood; softmax
  and the log fused into one overflow-safe function.
- gradient descent: repeated small steps against each parameter's
  gradient; the only training method we use.
- autograd engine: code that computes every parameter's gradient for any
  computation, automatically; micrograd builds one (phase 1).
- smoothing: counting a 1 before a 0 so nothing is impossible by
  construction.
- held-out: a corpus slice kept out of training; perplexity is measured
  there.
- gradient: how much the loss changes if one parameter is nudged.
- learning rate: the step size applied to parameters using their gradients.
- epoch: one full pass over the training data.
- checkpoint: a snapshot of all parameters, written by a CheckpointStore.
- training run: one execution of the training loop over a corpus.
- perplexity: loss converted to "how many equally likely choices per
  token"; lower is better; our target metric (ADR-0004).

## Generating

- stage: one selectable model in the chat's stage menu; it names itself,
  describes itself, and knows how to build a session (composition wires
  the concrete ones).
- sample: pick the next token from the predicted distribution.
- temperature: flattens or sharpens the distribution before sampling.
- top-k: sample only from the k most likely tokens.
