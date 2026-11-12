from functools import partial

from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.checkpoints import restore, snapshot_of
from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_neural_bigram,
    train_transformer_lm,
)
from murrai.presentation.chat.panel import Command
from murrai.presentation.chat.session import ChatSession


def _trained_chat_session(builder, directory, seed, store, on_event=None):
    bus = ListenerBus()
    if on_event is not None:
        bus.subscribe(on_event)
    try:
        result = builder(directory, seed, bus=bus)
    finally:
        if on_event is not None:
            bus.unsubscribe(on_event)
    if store is not None:
        store.save(result.model.kind, snapshot_of(result.model))
    return ChatSession(result.model, seed)


def counted_from_directory(directory, seed=0, bus=None):
    return train_bigram(DirTextCorpus(directory), holdout=0.1, bus=bus)


def neural_from_directory(directory, seed=0, steps=2000, batch=64, bus=None):
    return train_neural_bigram(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        learning_rate=3.0,
        bus=bus,
    )


def mlp_from_directory(directory, seed=0, steps=10000, batch=256, bus=None):
    return train_mlp_lm(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        learning_rate=0.5,
        patience=1000,
        window=8,
        dimension=32,
        hidden=256,
        bus=bus,
    )


def gpt_from_directory(directory, seed=0, steps=2000, batch=256, bus=None):
    return train_transformer_lm(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        depth=4,
        dimension=64,
        max_time=64,
        bus=bus,
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
):
    commands = []
    if store is not None:
        for name in store.names():
            commands.append(
                Command(
                    f"load {name}",
                    "pick up this trained model",
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
                ),
                trains=True,
            ),
        ],
    )
    return commands
