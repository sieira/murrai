import itertools
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from murrai.application.ports import Corpus, EventBus, TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM


@dataclass
class TrainingResult:
    model: Bigram
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float


@dataclass
class NeuralTrainingResult:
    model: NeuralBigram
    loss_curve: list[float]


class _StageBus:
    # The domain reports stage-free Progress; the application names the
    # stage on the way out (ADR-0005).
    def __init__(self, bus, phase):
        self._bus = bus
        self._phase = phase

    def emit(self, event):
        self._bus.emit(
            TrainingEvent(
                step=event.step,
                total_steps=event.total_steps,
                phase=self._phase,
            ),
        )


def _stage_bus(bus, phase):
    return None if bus is None else _StageBus(bus, phase)


def train_bigram(
    corpus: Corpus,
    holdout: float,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
) -> TrainingResult:
    text = corpus.text()
    cut = int(len(text) * (1 - holdout))
    train, held_out = text[:cut], text[cut:]

    tokenizer = TokenizerEncoderDecoder().fit(
        train,
        merges,
        bus=_stage_bus(bus, "tokenising"),
    )
    model = Bigram.fit(
        train,
        tokenizer,
        bus=_stage_bus(bus, "counting"),
    )
    if keeper is not None:
        keeper(model)
    return TrainingResult(
        model=model,
        train_chars=len(train),
        held_out_chars=len(held_out),
        held_out_perplexity=model.perplexity(held_out),
    )


def train_neural_bigram(
    train,
    steps,
    batch,
    seed,
    learning_rate=0.5,
    final_learning_rate=0.1,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
):
    tokenizer = TokenizerEncoderDecoder().fit(
        train,
        merges,
        bus=_stage_bus(bus, "tokenising"),
    )
    model = NeuralBigram(tokenizer)
    ids = [
        tokenizer.vocab_size,
        *tokenizer.encode(train, bus=_stage_bus(bus, "encoding")),
    ]
    pairs = [(prev, nxt) for prev, nxt in itertools.pairwise(ids)]

    rng = random.Random(seed)
    loss_curve = []
    for step in range(steps):
        rate = learning_rate + (final_learning_rate - learning_rate) * step / (
            steps - 1
        )
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(batch)]
        prevs = [prev for prev, _ in sample]
        nexts = [nxt for _, nxt in sample]
        loss = model.batch_loss(prevs, nexts)
        loss.backward()
        model.update(rate)
        loss_curve.append(loss.data)
        if bus is not None:
            bus.emit(TrainingEvent(step=step + 1, total_steps=steps))
    if keeper is not None:
        keeper(model)
    return NeuralTrainingResult(model=model, loss_curve=loss_curve)


def training_device(prefer_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plateau_rate(since_best, rate, patience, shrink):
    if since_best >= patience:
        return rate * shrink
    return rate


@dataclass
class MlpTrainingResult:
    model: MlpLM
    loss_curve: list[float]
    rate_curve: list[float]
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float


def train_mlp_lm(
    text,
    steps,
    batch,
    seed,
    window=4,
    dimension=16,
    hidden=128,
    learning_rate=1.0,
    patience=200,
    shrink=0.5,
    holdout=0.1,
    device=None,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
):
    device = torch.device(device) if device is not None else torch.device("cpu")
    cut = int(len(text) * (1 - holdout))
    train, held_out = text[:cut], text[cut:]

    tokenizer = TokenizerEncoderDecoder().fit(
        train,
        merges,
        bus=_stage_bus(bus, "tokenising"),
    )
    torch.manual_seed(seed)
    model = MlpLM(tokenizer, window=window, dimension=dimension, hidden=hidden).to(
        device,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ids = [
        model.start,
        *tokenizer.encode(train, bus=_stage_bus(bus, "encoding")),
    ]
    contexts = [ids[max(0, i - window) : i] for i in range(1, len(ids))]
    targets = ids[1:]
    pairs = len(targets)

    rng = random.Random(seed)
    loss_curve, rate_curve = [], []
    best = float("inf")
    best_step = 0
    rate = learning_rate
    for step in range(steps):
        chosen = [rng.randrange(pairs) for _ in range(batch)]
        batch_targets = torch.tensor(
            [targets[i] for i in chosen],
            dtype=torch.long,
            device=device,
        )
        loss = F.cross_entropy(
            model.forward_batch([contexts[i] for i in chosen]),
            batch_targets,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        value = float(loss.detach())
        best_step = step if value < best else best_step
        best = min(best, value)
        rate = plateau_rate(step - best_step, rate, patience, shrink)
        optimizer.param_groups[0]["lr"] = rate
        loss_curve.append(value)
        rate_curve.append(rate)
        if bus is not None:
            bus.emit(TrainingEvent(step=step + 1, total_steps=steps))

    if keeper is not None:
        keeper(model)
    return MlpTrainingResult(
        model=model,
        loss_curve=loss_curve,
        rate_curve=rate_curve,
        train_chars=len(train),
        held_out_chars=len(held_out),
        held_out_perplexity=model.perplexity(held_out),
    )


@dataclass
class TransformerTrainingResult:
    model: TransformerLM
    loss_curve: list[float]
    rate_curve: list[float]
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float


def train_transformer_lm(
    text,
    steps,
    batch,
    seed,
    depth=4,
    dimension=64,
    max_time=64,
    heads=1,
    learning_rate=1e-3,
    patience=200,
    shrink=0.5,
    holdout=0.1,
    device=None,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
):
    device = torch.device(device) if device is not None else torch.device("cpu")
    cut = int(len(text) * (1 - holdout))
    train, held_out = text[:cut], text[cut:]

    tokenizer = TokenizerEncoderDecoder().fit(
        train,
        merges,
        bus=_stage_bus(bus, "tokenising"),
    )
    torch.manual_seed(seed)
    model = TransformerLM(
        tokenizer,
        depth=depth,
        dimension=dimension,
        max_time=max_time,
        heads=heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ids = [
        model.start,
        *tokenizer.encode(train, bus=_stage_bus(bus, "encoding")),
    ]
    stream = torch.tensor(ids, dtype=torch.long, device=device)
    if len(stream) < 2:
        raise ValueError("corpus encodes to fewer than two tokens")
    window = min(max_time, len(stream) - 1)
    span = torch.arange(window, device=device)
    last_start = len(stream) - window - 1

    rng = random.Random(seed)
    loss_curve, rate_curve = [], []
    best = float("inf")
    best_step = 0
    rate = learning_rate
    for step in range(steps):
        chosen = torch.tensor(
            [rng.randrange(max(last_start, 1)) for _ in range(batch)],
            device=device,
        )
        rows = stream[chosen[:, None] + span]
        targets = stream[chosen[:, None] + span + 1]
        logits = model(rows)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        value = float(loss.detach())
        best_step = step if value < best else best_step
        best = min(best, value)
        rate = plateau_rate(step - best_step, rate, patience, shrink)
        optimizer.param_groups[0]["lr"] = rate
        loss_curve.append(value)
        rate_curve.append(rate)
        if bus is not None:
            bus.emit(TrainingEvent(step=step + 1, total_steps=steps))

    if keeper is not None:
        keeper(model)
    return TransformerTrainingResult(
        model=model,
        loss_curve=loss_curve,
        rate_curve=rate_curve,
        train_chars=len(train),
        held_out_chars=len(held_out),
        held_out_perplexity=model.perplexity(held_out),
    )
