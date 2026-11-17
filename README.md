# murrAI

## What this is

murrAI is a conversational language model built and trained from scratch, on
one machine (an Apple M4 with a Metal GPU), to learn how LLMs work on the
inside. It is a learning project for personal knowledge, not a product, and
it does not aim to be state of the art. [SPEC.md](SPEC.md) holds the phase
plan and the full target. The target in short
([ADR-0004](doc/adr/0004-target-gpt2-small.md)): a 10-50M parameter model
that matches or beats [GPT-2-small](https://openai.com/index/better-language-models/)
per parameter within its own domain. The chat output stays crude on purpose.

## Where it stands

What works today:

- Four model kinds train on the same corpus, and all of them generate, score
  text, and save: the counted bigram, the neural bigram, a windowed MLP, and
  a GPT-style transformer with multi-head attention.
- A byte-level BPE tokenizer turns any string into token ids and back
  ([ADR-0008](doc/adr/0008-byte-level-bpe-tokenizer.md)).
- Checkpoints hold the weights, the merge table, and a record of the run, and
  a new training can continue from one
  ([ADR-0009](doc/adr/0009-what-a-checkpoint-holds.md)).
- Training reports progress as events and runs in its own process, so the
  interface stays live while it works
  ([ADR-0005](doc/adr/0005-training-progress-events.md)).
- A full-screen chat trains, loads, and switches models
  ([ADR-0006](doc/adr/0006-chat-owns-the-whole-screen.md)) and records what it
  did in log lanes ([ADR-0007](doc/adr/0007-chat-log-lanes.md)).
- A tool enforces the layer map, not review alone
  ([ADR-0002](doc/adr/0002-pragmatic-ddd-and-hexagonal.md),
  [ADR-0003](doc/adr/0003-machine-enforced-boundaries.md)).

The study notes below write up the reasoning behind each step;
[doc/study/history.md](doc/study/history.md) covers where the ideas came
from. What comes next lives in [SPEC.md](SPEC.md).

## Try it

You need [uv](https://docs.astral.sh/uv/), Python 3.14 or newer,
[torch](https://pytorch.org/) 2.5 or newer, and
[prompt-toolkit](https://python-prompt-toolkit.readthedocs.io/).

```bash
make setup    # install dependencies into .venv
make corpus   # download the training corpus into data/corpus/
make chat     # start the chat
```

`make corpus` downloads 37 public-domain books from
[Project Gutenberg](https://www.gutenberg.org/) and caches them in
`data/corpus/`. `make chat` opens the chat; with no trained model yet it
starts with the menu focused, where you pick a training command and wait
for it to finish. The chat shows the story so far, offers the model's
continuation as ghost text after your cursor (Tab takes it), and lists the
odds the model gives its next few tokens.

## Commands

Every target in the [Makefile](Makefile):

| Command | What it does |
|---|---|
| `make setup` | creates `.venv` and installs the dependencies |
| `make check` | lint, boundaries, and tests together; the session-end gate |
| `make test` | runs the full pytest suite |
| `make test-one SCRIPT=<text>` | runs the tests whose name contains `<text>`, for iterating |
| `make lint` | [ruff](https://docs.astral.sh/ruff/) checks and formats `murrai/`, `tests/`, `scripts/` |
| `make boundaries` | [import-linter](https://import-linter.readthedocs.io/) checks the layer map |
| `make bench` | times matrix multiplication on the CPU and the GPU, as a check of the machine |
| `make bench-train` | times the MLP and GPT training runs on the corpus; takes step counts as arguments |
| `make corpus` | downloads the corpus |
| `make chat` | starts the chat |

## How it is built

The code lives in five layers inside `murrai/`, after
[pragmatic DDD in a hexagon](doc/adr/0002-pragmatic-ddd-and-hexagonal.md):

- `domain/` is pure math: the tokenizer, the models, sampling, and a
  hand-written autograd engine. Torch tensor operations are allowed; torch
  file I/O, network access, and filesystem writes are not.
- `application/` orchestrates: training runs, checkpoint snapshots and
  records. It depends on domain and on ports, the interfaces the outer
  layers implement.
- `adapters/` implement the ports: corpus and text downloads, checkpoint
  files, the event bus.
- `presentation/` renders state and computes no math. The chat is a
  separable unit inside it, `presentation/chat/`, which touches the core
  only through the `LanguageModel` port.
- `composition/` wires it all together: it builds the sessions, the
  commands, and the checkpoint records.

The dependency map is written down in [doc/ddd/layers.md](doc/ddd/layers.md),
the project's vocabulary in [doc/ddd/glossary.md](doc/ddd/glossary.md). A
tool enforces the map (`make boundaries`), so a layer cannot quietly start
importing another ([ADR-0003](doc/adr/0003-machine-enforced-boundaries.md)).

## The model and its records

- **Target.** The pass-or-fail goal, and the published evidence behind it,
  is [ADR-0004](doc/adr/0004-target-gpt2-small.md).
- **Training events.** Trainers report progress as plain numbers through a
  port, the application names the stages, and each training runs in its own
  process; only the finished checkpoint's name crosses back
  ([ADR-0005](doc/adr/0005-training-progress-events.md),
  [murrai/application/runs.py](murrai/application/runs.py)).
- **Chat screen.** One full-screen prompt-toolkit application owns the
  terminal until you quit ([ADR-0006](doc/adr/0006-chat-owns-the-whole-screen.md),
  [murrai/presentation/chat/](murrai/presentation/chat/)).
- **Log lanes.** The chat records what it did as timestamped events and
  appends them to `chat.log`
  ([ADR-0007](doc/adr/0007-chat-log-lanes.md),
  [murrai/presentation/chat/lanes.py](murrai/presentation/chat/lanes.py)).
- **Tokenizer.** Text becomes ids by byte-level BPE, in the style of
  [minbpe](https://github.com/karpathy/minbpe): the base vocabulary is all
  256 byte values, so every string encodes, and each learned merge turns the
  most frequent adjacent pair into one new id
  ([ADR-0008](doc/adr/0008-byte-level-bpe-tokenizer.md),
  [murrai/domain/tokenizer.py](murrai/domain/tokenizer.py)).
- **Checkpoint.** A checkpoint holds the model kind, the tokenizer's merge
  table, the weights, and a record of the run: when it was saved, on what
  corpus, with what seed, and the loss and perplexity the run measured. A
  checkpoint is also a starting point: a new training can rebuild its
  weights and keep going
  ([ADR-0009](doc/adr/0009-what-a-checkpoint-holds.md),
  [murrai/application/checkpoints.py](murrai/application/checkpoints.py),
  [murrai/adapters/checkpoint.py](murrai/adapters/checkpoint.py)).

## Repository map

```
murrai/     the five layers (see How it is built above)
tests/      the pytest suite, one file per behavior
scripts/    runnable demos and measurements
doc/        adr/ decisions, ddd/ layers and glossary, study/ theory notes
```

The scripts:

- [scripts/chat.py](scripts/chat.py) starts the chat
- [scripts/fetch_gutenberg.py](scripts/fetch_gutenberg.py) downloads the
  corpus books
- [scripts/fetch_tinystories.py](scripts/fetch_tinystories.py) downloads a
  2M-character slice of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- [scripts/mps_sanity.py](scripts/mps_sanity.py) times matrix multiplication
  per device (this is `make bench`)
- [scripts/autograd_bridge.py](scripts/autograd_bridge.py) trains the neural
  bigram and measures how close it lands to the counted bigram's table
- [scripts/train_bigram_demo.py](scripts/train_bigram_demo.py) trains the
  counted bigram on a directory and prints perplexity and sample text
- [scripts/train_mlp_demo.py](scripts/train_mlp_demo.py) trains the windowed
  MLP the same way
- [scripts/train_attention_demo.py](scripts/train_attention_demo.py) trains
  the bigram, the MLP, and the GPT on one corpus and prints a comparison
  table plus samples
- [scripts/train_heads_demo.py](scripts/train_heads_demo.py) trains the GPT
  with chosen heads, depth, and steps, and can save the result as a
  checkpoint
- [scripts/train_bench.py](scripts/train_bench.py) times the training runs
  (this is `make bench-train`)
- [scripts/tokenizer_density.py](scripts/tokenizer_density.py) measures how
  densely the tokenizer encodes prose at a given merge count
- [scripts/tokenizer_fit_speed.py](scripts/tokenizer_fit_speed.py) times
  tokenizer fitting and encoding on the corpus

## Reading

The architecture decisions, in [doc/adr/](doc/adr/), oldest first:

1. [Record architecture decisions](doc/adr/0001-record-architecture-decisions.md)
2. [Pragmatic DDD and hexagonal architecture for murrAI](doc/adr/0002-pragmatic-ddd-and-hexagonal.md)
3. [Machine-enforced boundaries](doc/adr/0003-machine-enforced-boundaries.md)
4. [Target: match or beat GPT-2-small per parameter](doc/adr/0004-target-gpt2-small.md)
5. [Training reports progress as events, from its own process](doc/adr/0005-training-progress-events.md)
6. [The chat is one full-screen application](doc/adr/0006-chat-owns-the-whole-screen.md)
7. [The chat records itself as events in log lanes](doc/adr/0007-chat-log-lanes.md)
8. [Text is tokenized by byte-level BPE](doc/adr/0008-byte-level-bpe-tokenizer.md)
9. [What a checkpoint holds](doc/adr/0009-what-a-checkpoint-holds.md)

The theory behind each phase, in [doc/study/](doc/study/):

- [How next-token prediction became reasoning](doc/study/history.md)
- [The autograd engine (micrograd)](doc/study/micrograd.md)
- [Bigram language models](doc/study/bigrams.md)
- [Universal approximation](doc/study/approximation.md)
- [The model's anatomy](doc/study/internals.md)
- [The MLP language model](doc/study/mlp.md)
- [Attention](doc/study/attention.md)
- [The transformer](doc/study/transformer.md)

Six of the study notes lean on Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
[nanoGPT](https://github.com/karpathy/nanoGPT), and
[minbpe](https://github.com/karpathy/minbpe); each note cites what it used.

## Working on this repo

The working rules live in [AGENTS.md](AGENTS.md): a failing test before every
change, decisions recorded as ADRs, one session to one branch and one pull
request. Work is tracked on the
[murrAI work board](https://github.com/users/sieira/projects/2).

## License

[GPL-3.0](LICENSE).
