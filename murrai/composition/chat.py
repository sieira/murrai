from dataclasses import dataclass
from typing import Any

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


def _trained_session(build, listener=None):
    bus = ListenerBus()
    if listener is not None:
        bus.subscribe(listener)
    try:
        return build(bus=bus)
    finally:
        if listener is not None:
            bus.unsubscribe(listener)


@dataclass(frozen=True)
class Retrain:
    builder: Any
    directory: Any
    seed: int
    steps: int
    store: Any

    def __call__(self, on_event=None):
        return _trained_session(
            lambda bus: self.builder(
                self.directory,
                self.seed,
                steps=self.steps,
                store=self.store,
                bus=bus,
            ),
            on_event,
        )


def chat_session_from_directory(directory, seed=0):
    result = train_bigram(DirTextCorpus(directory), holdout=0.1)
    return ChatSession(result.model, seed)


def neural_chat_session_from_directory(
    directory,
    seed=0,
    steps=2000,
    batch=64,
    store=None,
    bus=None,
):
    result = train_neural_bigram(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        learning_rate=3.0,
        bus=bus,
    )
    if store is not None:
        store.save(result.model.kind, snapshot_of(result.model))
    return ChatSession(result.model, seed)


def mlp_chat_session_from_directory(
    directory,
    seed=0,
    steps=10000,
    batch=256,
    store=None,
    bus=None,
):
    result = train_mlp_lm(
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
    if store is not None:
        store.save(result.model.kind, snapshot_of(result.model))
    return ChatSession(result.model, seed)


def char_gpt_chat_session_from_directory(
    directory,
    seed=0,
    steps=2000,
    batch=256,
    store=None,
    bus=None,
):
    result = train_transformer_lm(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        depth=4,
        dimension=64,
        max_time=64,
        bus=bus,
    )
    if store is not None:
        store.save(result.model.kind, snapshot_of(result.model))
    return ChatSession(result.model, seed)


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
                lambda: chat_session_from_directory(directory, seed),
            ),
            Command(
                "retrain neural bigram",
                "same memory, trained by gradient descent",
                Retrain(
                    neural_chat_session_from_directory,
                    directory,
                    seed,
                    neural_steps,
                    store,
                ),
                trains=True,
            ),
            Command(
                "retrain windowed mlp",
                "eight chars of memory through a hidden layer",
                Retrain(
                    mlp_chat_session_from_directory,
                    directory,
                    seed,
                    mlp_steps,
                    store,
                ),
                trains=True,
            ),
            Command(
                "retrain char gpt",
                "attention over the last sixty-four chars",
                Retrain(
                    char_gpt_chat_session_from_directory,
                    directory,
                    seed,
                    gpt_steps,
                    store,
                ),
                trains=True,
            ),
        ],
    )
    return commands
