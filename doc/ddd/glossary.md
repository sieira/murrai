# Glossary

The ubiquitous language. Identifiers in code, tests, and docs use these
terms. Entries are added at decision time, one plain line each. This file
binds (AGENTS.md header).

## Numbers and structures

- tensor: a multi-dimensional array of numbers; everything in murrAI is one.
- parameter: one learned number inside the model; murrAI targets 10-50M.
- token: one integer the model reads or predicts; text becomes tokens.
- tokenizer: the pair of functions turning text into tokens and back.
- BPE: byte pair encoding; the algorithm our tokenizer learns as merges of
  frequent pairs.
- embedding: a learned vector standing for one token.
- context window: how many previous tokens the model can see at once.
- logits: raw scores, one per vocabulary entry, before normalization.
- softmax: turns logits into probabilities that sum to one.
- attention head: one learned pattern of "which tokens look at which".
- batch: a group of sequences trained on at once.

## Training

- loss: one number measuring how wrong the predictions are; training
  lowers it.
- gradient: how much the loss changes if one parameter is nudged.
- learning rate: the step size applied to parameters using their gradients.
- epoch: one full pass over the training data.
- checkpoint: a snapshot of all parameters, written by a CheckpointStore.
- training run: one execution of the training loop over a corpus.
- perplexity: loss converted to "how many equally likely choices per
  token"; lower is better; our target metric (ADR-0004).

## Generating

- sample: pick the next token from the predicted distribution.
- temperature: flattens or sharpens the distribution before sampling.
- top-k: sample only from the k most likely tokens.
