import time
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial

from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.checkpoints import restore, snapshot_of
from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_neural_bigram,
    train_transformer_lm,
    training_device,
)
from murrai.presentation.chat.panel import Command
from murrai.presentation.chat.session import ChatSession


def _saved_at(clock):
    return datetime.fromtimestamp(clock(), tz=UTC).isoformat(timespec="seconds")


def _short_time(stamp):
    return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")


def _result_info(result):
    info = {}
    curve = getattr(result, "loss_curve", [])
    if curve:
        info["final_loss"] = curve[-1]
    perplexity = getattr(result, "held_out_perplexity", None)
    if perplexity is not None:
        info["held_out_perplexity"] = perplexity
    return info


def _load_description(info):
    if not info:
        return "pick up this trained model"
    parts = []
    if "saved_at" in info:
        parts.append(f"saved {_short_time(info['saved_at'])}")
    if "steps" in info:
        parts.append(f"{info['steps']} steps")
    if "final_loss" in info:
        parts.append(f"loss {info['final_loss']:.2f}")
    if "held_out_perplexity" in info:
        parts.append(f"ppl {info['held_out_perplexity']:.1f}")
    return " · ".join(parts)


def _trained_chat_session(
    builder,
    directory,
    seed,
    store,
    clock=time.time,
    on_event=None,
):
    bus = ListenerBus()
    if on_event is not None:
        bus.subscribe(on_event)
    kept = None

    def keep(model, settings):
        nonlocal kept
        if store is None:
            return
        kept = snapshot_of(
            model,
            info={
                "saved_at": _saved_at(clock),
                "corpus": str(directory),
                "seed": seed,
                **settings,
            },
        )
        store.save(model.kind, kept)

    try:
        result = builder(directory, seed, bus=bus, keeper=keep)
    finally:
        if on_event is not None:
            bus.unsubscribe(on_event)

    if kept is not None:
        store.save(
            result.model.kind,
            replace(kept, info={**kept.info, **_result_info(result)}),
        )
    return ChatSession(result.model, seed)


def counted_from_directory(directory, seed=0, bus=None, keeper=None):
    def keep(model):
        if keeper is not None:
            keeper(model, {})

    return train_bigram(
        DirTextCorpus(directory),
        holdout=0.1,
        bus=bus,
        keeper=keep,
    )


def neural_from_directory(
    directory,
    seed=0,
    steps=2000,
    batch=64,
    bus=None,
    keeper=None,
):
    settings = {"steps": steps, "batch": batch, "learning_rate": 3.0}

    def keep(model):
        if keeper is not None:
            keeper(model, settings)

    return train_neural_bigram(
        DirTextCorpus(directory).text(),
        seed=seed,
        bus=bus,
        keeper=keep,
        **settings,
    )


def mlp_from_directory(
    directory,
    seed=0,
    steps=10000,
    batch=256,
    bus=None,
    keeper=None,
):
    settings = {"steps": steps, "batch": batch, "learning_rate": 0.5}

    def keep(model):
        if keeper is not None:
            keeper(model, settings)

    return train_mlp_lm(
        DirTextCorpus(directory).text(),
        seed=seed,
        patience=1000,
        window=8,
        dimension=32,
        hidden=256,
        device=training_device(prefer_mps=False),
        bus=bus,
        keeper=keep,
        **settings,
    )


def gpt_from_directory(
    directory,
    seed=0,
    steps=2000,
    batch=256,
    bus=None,
    keeper=None,
):
    settings = {"steps": steps, "batch": batch, "learning_rate": 1e-3}

    def keep(model):
        if keeper is not None:
            keeper(model, settings)

    return train_transformer_lm(
        DirTextCorpus(directory).text(),
        seed=seed,
        depth=4,
        dimension=64,
        max_time=64,
        device=training_device(prefer_mps=True),
        bus=bus,
        keeper=keep,
        **settings,
    )


def load_checkpoint_session(store, seed=0, name=None):
    if name is None:
        name = store.names()[-1]
    model = restore(store.load(name))
    return ChatSession(model, seed)


def chat_commands(
    directory,
    seed=0,
    store=None,
    neural_steps=2000,
    mlp_steps=10000,
    gpt_steps=2000,
    clock=time.time,
):
    commands = []
    if store is not None:
        for name in store.names():
            commands.append(
                Command(
                    f"load {name}",
                    _load_description(store.load(name).info),
                    lambda name=name: load_checkpoint_session(store, seed, name),
                ),
            )
    commands.extend(
        [
            Command(
                "counted bigram",
                "next-char odds tallied straight from the corpus",
                partial(
                    _trained_chat_session,
                    counted_from_directory,
                    directory,
                    seed,
                    store,
                    clock,
                ),
                trains=True,
            ),
            Command(
                "retrain neural bigram",
                "same memory, trained by gradient descent",
                partial(
                    _trained_chat_session,
                    partial(neural_from_directory, steps=neural_steps),
                    directory,
                    seed,
                    store,
                    clock,
                ),
                trains=True,
            ),
            Command(
                "retrain windowed mlp",
                "eight chars of memory through a hidden layer",
                partial(
                    _trained_chat_session,
                    partial(mlp_from_directory, steps=mlp_steps),
                    directory,
                    seed,
                    store,
                    clock,
                ),
                trains=True,
            ),
            Command(
                "retrain char gpt",
                "attention over the last sixty-four chars",
                partial(
                    _trained_chat_session,
                    partial(gpt_from_directory, steps=gpt_steps),
                    directory,
                    seed,
                    store,
                    clock,
                ),
                trains=True,
            ),
        ],
    )
    return commands
