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
from murrai.presentation.chat.form import FRESH, START_FROM, Field
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


def _trained_chat_session(run, directory, seed, store, clock, values, on_event=None):
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
        result = run(directory, seed, store, values, bus=bus, keeper=keep)
    finally:
        if on_event is not None:
            bus.unsubscribe(on_event)

    if kept is not None:
        store.save(
            result.model.kind,
            replace(kept, info={**kept.info, **_result_info(result)}),
        )
    return ChatSession(result.model, seed)


def _keep_with(keeper, settings):
    def keep(model):
        if keeper is not None:
            keeper(model, settings)

    return keep


def _start_snapshot(store, values):
    if store is None:
        return None
    name = values.get(START_FROM, FRESH)
    if name == FRESH:
        return None
    return store.load(name)


def _device(values, prefer_mps):
    return training_device(
        prefer_mps=prefer_mps,
        choice=values.get("device", "auto"),
    )


def _run_counted(directory, seed, store, values, bus=None, keeper=None):
    return train_bigram(
        DirTextCorpus(directory),
        holdout=values["holdout"],
        bus=bus,
        keeper=_keep_with(keeper, {}),
    )


def _run_neural(directory, seed, store, values, bus=None, keeper=None):
    settings = {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
    }
    return train_neural_bigram(
        DirTextCorpus(directory).text(),
        steps=values["steps"],
        batch=values["batch"],
        learning_rate=values["learning_rate"],
        final_learning_rate=values["final_learning_rate"],
        seed=values.get("seed", seed),
        start=_start_snapshot(store, values),
        bus=bus,
        keeper=_keep_with(keeper, settings),
    )


def _run_mlp(directory, seed, store, values, bus=None, keeper=None):
    settings = {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
    }
    start = _start_snapshot(store, values)
    kwargs = {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
        "patience": values["patience"],
        "shrink": values["shrink"],
        "holdout": values["holdout"],
        "device": _device(values, prefer_mps=False),
        "seed": values.get("seed", seed),
        "start": start,
    }
    if start is None:
        kwargs.update(
            window=values["window"],
            dimension=values["dimension"],
            hidden=values["hidden"],
        )
    return train_mlp_lm(
        DirTextCorpus(directory).text(),
        bus=bus,
        keeper=_keep_with(keeper, settings),
        **kwargs,
    )


def _run_gpt(directory, seed, store, values, bus=None, keeper=None):
    settings = {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
    }
    start = _start_snapshot(store, values)
    kwargs = {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
        "patience": values["patience"],
        "shrink": values["shrink"],
        "holdout": values["holdout"],
        "device": _device(values, prefer_mps=True),
        "seed": values.get("seed", seed),
        "start": start,
    }
    if start is None:
        kwargs.update(
            depth=values["depth"],
            dimension=values["dimension"],
            max_time=values["max_time"],
            heads=values["heads"],
        )
    return train_transformer_lm(
        DirTextCorpus(directory).text(),
        bus=bus,
        keeper=_keep_with(keeper, settings),
        **kwargs,
    )


def _bake_counted(directory, seed, store, clock, values):
    return Command(
        "counted bigram",
        "next-char odds tallied straight from the corpus",
        partial(
            _trained_chat_session,
            _run_counted,
            directory,
            seed,
            store,
            clock,
            values,
        ),
        trains=True,
        given=values,
    )


def _bake_neural(directory, seed, store, clock, values):
    return Command(
        "retrain neural bigram",
        "same memory, trained by gradient descent",
        partial(
            _trained_chat_session,
            _run_neural,
            directory,
            seed,
            store,
            clock,
            values,
        ),
        trains=True,
        given=values,
    )


def _bake_mlp(directory, seed, store, clock, values):
    return Command(
        "retrain windowed mlp",
        "eight chars of memory through a hidden layer",
        partial(
            _trained_chat_session,
            _run_mlp,
            directory,
            seed,
            store,
            clock,
            values,
        ),
        trains=True,
        given=values,
    )


def _bake_gpt(directory, seed, store, clock, values):
    return Command(
        "retrain char gpt",
        "attention over the last sixty-four chars",
        partial(
            _trained_chat_session,
            _run_gpt,
            directory,
            seed,
            store,
            clock,
            values,
        ),
        trains=True,
        given=values,
    )


def _holdout_field():
    return Field("holdout", "holdout", "float", 0.1, check="from zero below one")


def _seed_field(seed):
    return Field("seed", "seed", "int", seed, check="whole number")


def _start_field(store):
    names = store.names() if store is not None else ()
    return Field(START_FROM, "start from", "choice", FRESH, choices=(FRESH, *names))


def _device_field(default):
    return Field(
        "device",
        "device",
        "choice",
        default,
        choices=("auto", "prefer gpu", "cpu"),
    )


def _counted_fields():
    return (_holdout_field(),)


def _neural_fields(seed, store, steps):
    return (
        Field("steps", "steps", "int", steps, check="whole above zero"),
        Field("batch", "batch", "int", 64, check="whole above zero"),
        Field(
            "learning_rate",
            "learning rate",
            "float",
            3.0,
            check="number above zero",
        ),
        Field(
            "final_learning_rate",
            "final learning rate",
            "float",
            0.1,
            check="number above zero",
        ),
        _seed_field(seed),
        _start_field(store),
    )


def _mlp_fields(seed, store, steps):
    return (
        Field("steps", "steps", "int", steps, check="whole above zero"),
        Field("batch", "batch", "int", 256, check="whole above zero"),
        Field(
            "learning_rate",
            "learning rate",
            "float",
            0.5,
            check="number above zero",
        ),
        Field("window", "window", "int", 8, check="whole above zero", shape=True),
        Field(
            "dimension",
            "dimension",
            "int",
            32,
            check="whole above zero",
            shape=True,
        ),
        Field("hidden", "hidden", "int", 256, check="whole above zero", shape=True),
        Field("patience", "patience", "int", 1000, check="whole zero or more"),
        Field("shrink", "shrink", "float", 0.5, check="between zero and one"),
        _holdout_field(),
        _seed_field(seed),
        _device_field("auto"),
        _start_field(store),
    )


def _gpt_fields(seed, store, steps):
    return (
        Field("steps", "steps", "int", steps, check="whole above zero"),
        Field("batch", "batch", "int", 256, check="whole above zero"),
        Field(
            "learning_rate",
            "learning rate",
            "float",
            0.001,
            check="number above zero",
        ),
        Field("depth", "depth", "int", 4, check="whole above zero", shape=True),
        Field(
            "dimension",
            "dimension",
            "int",
            64,
            check="whole above zero",
            shape=True,
        ),
        Field("heads", "heads", "int", 1, check="whole above zero", shape=True),
        Field(
            "max_time",
            "context length",
            "int",
            64,
            check="whole above zero",
            shape=True,
        ),
        Field("patience", "patience", "int", 200, check="whole zero or more"),
        Field("shrink", "shrink", "float", 0.5, check="between zero and one"),
        _holdout_field(),
        _seed_field(seed),
        _device_field("auto"),
        _start_field(store),
    )


def _training_command(bake, directory, seed, store, clock, fields):
    defaults = {field.name: field.default for field in fields}
    baked = bake(directory, seed, store, clock, defaults)
    return Command(
        baked.name,
        baked.description,
        baked.build,
        trains=True,
        fields=fields,
        bake=partial(bake, directory, seed, store, clock),
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
            _training_command(
                _bake_counted,
                directory,
                seed,
                store,
                clock,
                _counted_fields(),
            ),
            _training_command(
                _bake_neural,
                directory,
                seed,
                store,
                clock,
                _neural_fields(seed, store, neural_steps),
            ),
            _training_command(
                _bake_mlp,
                directory,
                seed,
                store,
                clock,
                _mlp_fields(seed, store, mlp_steps),
            ),
            _training_command(
                _bake_gpt,
                directory,
                seed,
                store,
                clock,
                _gpt_fields(seed, store, gpt_steps),
            ),
        ],
    )
    return commands
