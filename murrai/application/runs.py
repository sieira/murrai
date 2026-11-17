import itertools
import math
import random
import time
from array import array
from dataclasses import dataclass
from itertools import batched

import torch
import torch.nn.functional as F

from murrai.application.checkpoints import Snapshot, run_snapshot
from murrai.application.ports import Corpus, EventBus, TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import ROWS_PER_PASS, TransformerLM


@dataclass
class TrainingResult:
    model: Bigram
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float | None


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
    held_out_text: str | None = None,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
) -> TrainingResult:
    train = corpus.text()
    held_out = held_out_text or ""

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
        held_out_perplexity=model.perplexity(held_out) if held_out else None,
    )


def _checkpoint_tokenizer(train, merges, checkpoint, bus):
    if checkpoint is None:
        tokenizer = TokenizerEncoderDecoder()
        return tokenizer, tokenizer.fit_encoded(
            train,
            merges,
            bus=_stage_bus(bus, "tokenising"),
        )
    tokenizer = TokenizerEncoderDecoder(checkpoint.merges)
    return tokenizer, tokenizer.encode(train, bus=_stage_bus(bus, "encoding"))


def _reject_contradictory_shape(checkpoint, **knobs):
    if checkpoint is None:
        return
    mismatches = [
        f"{name}={value}"
        for name, value in knobs.items()
        if value is not None and value != checkpoint.state.get(name, value)
    ]
    if mismatches:
        raise ValueError(
            f"a checkpoint keeps its own shape; pass no {', '.join(mismatches)}",
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
    checkpoint: Snapshot | None = None,
):
    tokenizer, train_ids = _checkpoint_tokenizer(train, merges, checkpoint, bus)
    _reject_contradictory_shape(checkpoint)
    if checkpoint is None:
        model = NeuralBigram(tokenizer)
    else:
        model = NeuralBigram.from_state(tokenizer, checkpoint.state)
    ids = [tokenizer.vocab_size, *train_ids]
    del train_ids
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
        loss_curve.append(loss.item())
        if bus is not None:
            bus.emit(TrainingEvent(step=step + 1, total_steps=steps))
    if keeper is not None:
        keeper(model)
    return NeuralTrainingResult(model=model, loss_curve=loss_curve)


def training_device(prefer_mps=True, choice="auto"):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "prefer gpu":
        prefer_mps = True
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
    held_out_perplexity: float | None


def train_mlp_lm(
    text,
    steps,
    batch,
    seed,
    window=None,
    dimension=None,
    hidden=None,
    learning_rate=1.0,
    patience=200,
    shrink=0.5,
    held_out_text: str | None = None,
    device=None,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
    checkpoint: Snapshot | None = None,
):
    device = torch.device(device) if device is not None else torch.device("cpu")
    train = text
    held_out = held_out_text or ""

    torch.manual_seed(seed)
    _reject_contradictory_shape(
        checkpoint,
        window=window,
        dimension=dimension,
        hidden=hidden,
    )
    tokenizer, train_ids = _checkpoint_tokenizer(train, merges, checkpoint, bus)
    if checkpoint is None:
        window = window or 4
        model = MlpLM(
            tokenizer,
            window=window,
            dimension=dimension or 16,
            hidden=hidden or 128,
        ).to(
            device,
        )
    else:
        model = MlpLM.from_state(tokenizer, checkpoint.state).to(device)
        window = model.window
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ids = [model.start, *train_ids]
    del train_ids
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
        held_out_perplexity=model.perplexity(held_out) if held_out else None,
    )


@dataclass
class TransformerTrainingResult:
    model: TransformerLM
    loss_curve: list[float]
    rate_curve: list[float]
    validation_curve: list[tuple[int, float]]
    held_out_perplexity: float | None
    steps: int
    tokens_per_second: float


def _draw_batch(stream, window, batch, generator):
    last_start = len(stream) - window - 1
    chosen = torch.randint(0, last_start + 1, (batch,), generator=generator)
    chosen = chosen.to(stream.device)
    span = torch.arange(window, device=stream.device)
    rows = stream[chosen[:, None] + span].long()
    return rows, stream[chosen[:, None] + span + 1].long()


def _window_loss(model, rows, targets):
    with torch.no_grad():
        logits = model(rows)
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1).long(),
            reduction="sum",
        ),
    )


def _stream_nll(model, stream, rows_per_pass=ROWS_PER_PASS):
    positions = len(stream) - 1
    if positions < 1:
        return None
    max_time = model.max_time
    full = positions // max_time
    total = 0.0
    count = 0
    offset = torch.arange(max_time, device=stream.device)
    for group in batched(range(full), rows_per_pass, strict=False):
        index = torch.tensor(group, device=stream.device)[:, None] * max_time + offset
        total += _window_loss(model, stream[index], stream[index + 1])
        count += len(group) * max_time
    if positions % max_time:
        start = full * max_time
        length = positions - start
        index = start + torch.arange(length, device=stream.device)
        total += _window_loss(
            model,
            stream[index].unsqueeze(0),
            stream[index + 1].unsqueeze(0),
        )
        count += length
    return total / count


def _fresh_run(learning_rate):
    return {
        "step": 0,
        "rate": learning_rate,
        "best": float("inf"),
        "best_step": 0,
        "train_seconds": 0.0,
        "optimizer": None,
        "loss_curve": [],
        "rate_curve": [],
        "validation_curve": [],
    }


def train_transformer_lm(
    text,
    steps,
    batch,
    seed,
    depth=None,
    dimension=None,
    max_time=None,
    heads=None,
    learning_rate=1e-3,
    patience=200,
    shrink=0.5,
    held_out_text=None,
    validate_every=None,
    checkpoint_every=None,
    snapshots=None,
    device=None,
    bus: EventBus | None = None,
    keeper=None,
    merges: int = 300,
    checkpoint: Snapshot | None = None,
):
    if steps < 1:
        raise ValueError("steps counts the steps this run adds; pass one or more")
    device = torch.device(device) if device is not None else torch.device("cpu")
    torch.manual_seed(seed)
    _reject_contradictory_shape(
        checkpoint,
        depth=depth,
        dimension=dimension,
        max_time=max_time,
        heads=heads,
    )
    held_text = held_out_text
    tokenizer, train_ids = _checkpoint_tokenizer(text, merges, checkpoint, bus)
    if checkpoint is None:
        max_time = max_time or 64
        model = TransformerLM(
            tokenizer,
            depth=depth or 4,
            dimension=dimension or 64,
            max_time=max_time,
            heads=heads or 1,
        ).to(device)
        run = _fresh_run(learning_rate)
    else:
        model = TransformerLM.from_state(tokenizer, checkpoint.state).to(device)
        max_time = model.max_time
        run = _fresh_run(learning_rate)
        run.update(checkpoint.state.get("run") or {})
        run["loss_curve"] = list(run["loss_curve"])
        run["rate_curve"] = list(run["rate_curve"])
        run["validation_curve"] = list(run["validation_curve"])
    start = run["step"]
    optimizer = torch.optim.Adam(model.parameters(), lr=run["rate"])
    if run["optimizer"] is not None:
        optimizer.load_state_dict(run["optimizer"])
        optimizer.param_groups[0]["lr"] = run["rate"]

    ids = array("i", (model.start,))
    ids.extend(train_ids)
    del train_ids
    stream = torch.frombuffer(memoryview(ids), dtype=torch.int32).to(device)
    del ids
    if len(stream) < 2:
        raise ValueError("corpus encodes to fewer than two tokens")
    held = None
    if held_text:
        held_ids = array("i", (model.start,))
        held_ids.extend(
            tokenizer.encode(held_text, bus=_stage_bus(bus, "encoding")),
        )
        held = torch.frombuffer(memoryview(held_ids), dtype=torch.int32).to(device)
        del held_ids
    window = min(max_time, len(stream) - 1)

    if held is not None and start == 0:
        _record_validation(model, held, run)

    generator = torch.Generator().manual_seed(seed)
    end = start + steps
    saved_step = None
    for step in range(start, end):
        generator.manual_seed(seed + step)
        mark = time.perf_counter()
        rows, targets = _draw_batch(stream, window, batch, generator)
        logits = model(rows)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        value = float(loss.detach())
        run["train_seconds"] += time.perf_counter() - mark
        run["best_step"] = step if value < run["best"] else run["best_step"]
        run["best"] = min(run["best"], value)
        run["rate"] = plateau_rate(
            step - run["best_step"],
            run["rate"],
            patience,
            shrink,
        )
        optimizer.param_groups[0]["lr"] = run["rate"]
        run["loss_curve"].append(value)
        run["rate_curve"].append(run["rate"])
        run["step"] = step + 1
        if bus is not None:
            bus.emit(TrainingEvent(step=step + 1, total_steps=end))
        if validate_every and held is not None and run["step"] % validate_every == 0:
            _record_validation(model, held, run)
        if (
            checkpoint_every
            and snapshots is not None
            and run["step"] % checkpoint_every == 0
        ):
            run["optimizer"] = optimizer.state_dict()
            saved_step = run["step"]
            snapshots(run_snapshot(model, run))

    if snapshots is not None and saved_step != run["step"]:
        run["optimizer"] = optimizer.state_dict()
        snapshots(run_snapshot(model, run))
    if keeper is not None:
        keeper(model)
    if held is not None and (
        not run["validation_curve"] or run["validation_curve"][-1][0] != run["step"]
    ):
        _record_validation(model, held, run)
    tokens = run["step"] * batch * window
    return TransformerTrainingResult(
        model=model,
        loss_curve=run["loss_curve"][start:],
        rate_curve=run["rate_curve"][start:],
        validation_curve=(
            run["validation_curve"]
            if start == 0
            else [entry for entry in run["validation_curve"] if entry[0] > start]
        ),
        held_out_perplexity=(
            math.exp(run["validation_curve"][-1][1])
            if run["validation_curve"]
            else None
        ),
        steps=run["step"],
        tokens_per_second=(
            tokens / run["train_seconds"] if run["train_seconds"] > 0 else 0.0
        ),
    )


def _record_validation(model, held, run):
    loss = _stream_nll(model, held)
    if loss is not None:
        run["validation_curve"].append((run["step"], loss))
