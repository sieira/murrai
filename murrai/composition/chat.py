from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.checkpoints import restore, snapshot_of
from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_neural_bigram,
    train_transformer_lm,
)
from murrai.presentation.chat.picker import Stage
from murrai.presentation.chat.progress import EtaLine
from murrai.presentation.chat.session import ChatSession


def _trained_session(build):
    bus = ListenerBus()
    eta = EtaLine()
    bus.subscribe(eta)
    eta.start()
    try:
        return build(bus=bus)
    finally:
        eta.stop()
        bus.unsubscribe(eta)


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


def load_checkpoint_session(store, seed=0):
    model = restore(store.load(store.names()[-1]))
    return ChatSession(model, seed)


def chat_stages(
    directory,
    seed=0,
    neural_steps=2000,
    mlp_steps=10000,
    gpt_steps=2000,
    store=None,
):
    stages = [
        Stage(
            "counted bigram",
            "next-char odds tallied straight from the corpus",
            lambda: chat_session_from_directory(directory, seed),
        ),
        Stage(
            "neural bigram",
            "same memory, trained by gradient descent",
            lambda: _trained_session(
                lambda bus: neural_chat_session_from_directory(
                    directory,
                    seed,
                    steps=neural_steps,
                    store=store,
                    bus=bus,
                ),
            ),
            trains=True,
        ),
        Stage(
            "windowed mlp",
            "eight chars of memory through a hidden layer",
            lambda: _trained_session(
                lambda bus: mlp_chat_session_from_directory(
                    directory,
                    seed,
                    steps=mlp_steps,
                    store=store,
                    bus=bus,
                ),
            ),
            trains=True,
        ),
        Stage(
            "char gpt",
            "attention over the last sixty-four chars",
            lambda: _trained_session(
                lambda bus: char_gpt_chat_session_from_directory(
                    directory,
                    seed,
                    steps=gpt_steps,
                    store=store,
                    bus=bus,
                ),
            ),
            trains=True,
        ),
    ]
    if store is not None and store.names():
        stages.append(
            Stage(
                "load from checkpoint",
                "pick up the most recent trained model",
                lambda: load_checkpoint_session(store, seed),
            ),
        )
    return stages
