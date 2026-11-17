import time
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from murrai.adapters.corpus import DirTextCorpus, TextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
    restore,
    snapshot_of,
)
from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_neural_bigram,
    train_transformer_lm,
    training_device,
)
from murrai.presentation.chat.commands import Command, FormCommand
from murrai.presentation.chat.form import FRESH, START_FROM, Field
from murrai.presentation.chat.session import ChatSession


def _saved_at(clock):
    return datetime.fromtimestamp(clock(), tz=UTC).isoformat(timespec="seconds")


def _short_time(stamp):
    return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")


def _result_info(result):
    curve = getattr(result, "loss_curve", [])
    perplexity = getattr(result, "held_out_perplexity", None)
    return curve[-1] if curve else None, perplexity


def _load_description(record):
    parts = []
    if record.settings:
        parts.append(f"saved {_short_time(record.settings.saved_at)}")
        if record.settings.corpus_chars is not None:
            parts.append(f"{record.settings.corpus_chars // 1000}k chars")
    if record.gradient:
        parts.append(f"{record.gradient.steps} steps")
        if record.gradient.final_loss is not None:
            parts.append(f"loss {record.gradient.final_loss:.2f}")
    if record.held_out_perplexity is not None:
        parts.append(f"ppl {record.held_out_perplexity:.1f}")
    return " · ".join(parts) or "pick up this trained model"


def _chosen_directory(directory, values):
    return Path(directory).parent / values["text"]


SIZES = {"all": None, "100k": 100000, "500k": 500000}

RETRAINABLE = frozenset({"neural-bigram", "windowed-mlp", "gpt"})


def _sliced_text(directory, values):
    text = DirTextCorpus(directory).text()
    cap = SIZES[values["size"]]
    return text if cap is None else text[:cap]


def _trained_chat_session(run, directory, seed, store, clock, values, on_event=None):
    directory = _chosen_directory(directory, values)
    text = _sliced_text(directory, values)
    cap = SIZES[values["size"]]
    bus = ListenerBus()
    if on_event is not None:
        bus.subscribe(on_event)
    kept = None
    name = None

    def keep(model, settings):
        nonlocal kept, name
        if store is None:
            return
        name = values["name"]
        store.save(name, snapshot_of(model))
        kept = CheckpointRecord(
            settings=RunSettings(
                saved_at=_saved_at(clock),
                corpus=str(directory),
                seed=seed,
                corpus_chars=len(text) if cap is not None else None,
            ),
            gradient=GradientRun(**settings) if settings else None,
        )
        store.save_record(name, kept)

    try:
        result = run(text, seed, store, values, bus=bus, keeper=keep)
        bus.flush()
    finally:
        if on_event is not None:
            bus.unsubscribe(on_event)

    if kept is not None:
        final_loss, perplexity = _result_info(result)
        if kept.gradient is not None and final_loss is not None:
            kept = replace(kept, gradient=replace(kept.gradient, final_loss=final_loss))
        if perplexity is not None:
            kept = replace(kept, held_out_perplexity=perplexity)
        store.save_record(name, kept)
    return ChatSession(result.model, seed, checkpoint=name)


def _keeping(keeper, settings):
    def keep(model):
        if keeper is not None:
            keeper(model, settings)

    return keep


def _checkpoint_snapshot(store, values):
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


def _settings(values):
    return {
        "steps": values["steps"],
        "batch": values["batch"],
        "learning_rate": values["learning_rate"],
    }


def _trainer_kwargs(values, seed, checkpoint, settings, extra, shape):
    kwargs = {
        **settings,
        "seed": values.get("seed", seed),
        "checkpoint": checkpoint,
        **extra,
    }
    if checkpoint is None:
        kwargs.update(merges=values["merges"])
        kwargs.update((name, values[name]) for name in shape)
    return kwargs


def _run_counted(text, seed, store, values, bus=None, keeper=None):
    return train_bigram(
        TextCorpus(text),
        holdout=values["holdout"],
        merges=values["merges"],
        bus=bus,
        keeper=_keeping(keeper, {}),
    )


def _run_neural(text, seed, store, values, bus=None, keeper=None):
    checkpoint = _checkpoint_snapshot(store, values)
    return train_neural_bigram(
        text,
        bus=bus,
        keeper=_keeping(keeper, _settings(values)),
        **_trainer_kwargs(
            values,
            seed,
            checkpoint,
            _settings(values),
            {"final_learning_rate": values["final_learning_rate"]},
            (),
        ),
    )


def _run_mlp(text, seed, store, values, bus=None, keeper=None):
    checkpoint = _checkpoint_snapshot(store, values)
    return train_mlp_lm(
        text,
        bus=bus,
        keeper=_keeping(keeper, _settings(values)),
        **_trainer_kwargs(
            values,
            seed,
            checkpoint,
            _settings(values),
            {
                "patience": values["patience"],
                "shrink": values["shrink"],
                "holdout": values["holdout"],
                "device": _device(values, prefer_mps=False),
            },
            ("window", "dimension", "hidden"),
        ),
    )


def _run_gpt(text, seed, store, values, bus=None, keeper=None):
    checkpoint = _checkpoint_snapshot(store, values)
    return train_transformer_lm(
        text,
        bus=bus,
        keeper=_keeping(keeper, _settings(values)),
        **_trainer_kwargs(
            values,
            seed,
            checkpoint,
            _settings(values),
            {
                "patience": values["patience"],
                "shrink": values["shrink"],
                "holdout": values["holdout"],
                "device": _device(values, prefer_mps=True),
            },
            ("depth", "dimension", "heads", "max_time"),
        ),
    )


def _name_field(prefix, directory):
    return Field(
        "name",
        "name",
        "text",
        f"{prefix}-{Path(directory).name}",
        check="at least one character",
    )


def _holdout_field():
    return Field("holdout", "holdout", "float", 0.1, check="from zero below one")


def _text_field(directory):
    parent = Path(directory).parent
    choices = (
        tuple(
            sorted(
                entry.name
                for entry in parent.iterdir()
                if entry.is_dir() and any(entry.glob("*.txt"))
            ),
        )
        if parent.exists()
        else ()
    )
    return Field(
        "text",
        "text",
        "choice",
        Path(directory).name,
        choices=choices or (Path(directory).name,),
    )


def _size_field():
    return Field("size", "size", "choice", "all", choices=("all", "100k", "500k"))


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


def _merges_field():
    return Field("merges", "merges", "int", 300, check="whole above zero", shape=True)


def _counted_fields(directory):
    return (
        _name_field("counted-bigram", directory),
        _text_field(directory),
        _size_field(),
        _holdout_field(),
        _merges_field(),
    )


def _neural_fields(directory, seed, steps):
    return (
        _name_field("neural-bigram", directory),
        _text_field(directory),
        _size_field(),
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
        _merges_field(),
    )


def _mlp_fields(directory, seed, steps):
    return (
        _name_field("windowed-mlp", directory),
        _text_field(directory),
        _size_field(),
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
        _merges_field(),
    )


def _gpt_fields(directory, seed, steps):
    return (
        _name_field("gpt", directory),
        _text_field(directory),
        _size_field(),
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
        _merges_field(),
    )


def _training_command(
    name,
    description,
    run,
    directory,
    seed,
    store,
    clock,
    fields,
    continues=False,
    pin=None,
):
    def bake(values):
        merged = {**values, **pin} if pin else values
        return Command(
            name,
            description,
            partial(
                _trained_chat_session,
                run,
                directory,
                seed,
                store,
                clock,
                merged,
            ),
            trains=True,
        )

    defaults = {field.name: field.default for field in fields}
    return FormCommand(
        name,
        description,
        bake(defaults).build,
        fields=fields,
        bake=bake,
        trains=True,
        continues=continues,
        pin=pin,
    )


def _sessions_group(stories, store, seed, session):
    loads = tuple(
        Command(
            f"load {name}",
            "pick up this saved story",
            partial(load_story_session, stories, store, seed, name),
        )
        for name in stories.names()
    )
    live = session() if session else None
    save = ()
    if live is not None:
        description = "write this story down to load it later"
        save = (
            FormCommand(
                "save story",
                description,
                partial(_save_story, stories, session, f"{live.checkpoint}-story"),
                fields=(
                    Field(
                        "name",
                        "name",
                        "text",
                        f"{live.checkpoint}-story",
                        check="at least one character",
                    ),
                ),
                bake=lambda values: Command(
                    "save story",
                    description,
                    partial(_save_story, stories, session, values["name"]),
                ),
            ),
        )
    return save + loads


def load_story_session(stories, store, seed, name):
    save = stories.load(name)
    model = restore(store.load(save.checkpoint))
    return ChatSession.from_save(save, model, seed)


def _save_story(stories, session, name):
    live = session()
    if live is None:
        return None
    stories.save(name, live.to_save())
    return live


def load_checkpoint_session(store, seed=0, name=None):
    if name is None:
        name = store.names()[-1]
    model = restore(store.load(name))
    return ChatSession(model, seed, checkpoint=name)


def chat_menu(
    directory,
    seed=0,
    store=None,
    stories=None,
    session=None,
    neural_steps=2000,
    mlp_steps=10000,
    gpt_steps=2000,
    clock=time.time,
):
    counted = _training_command(
        "counted bigram",
        "next-char odds tallied straight from the corpus",
        _run_counted,
        directory,
        seed,
        store,
        clock,
        _counted_fields(directory),
    )
    retrain_kinds = {
        "neural-bigram": (
            "same memory, trained by gradient descent",
            _run_neural,
            lambda: _neural_fields(directory, seed, neural_steps),
        ),
        "windowed-mlp": (
            "a short token window through a hidden layer",
            _run_mlp,
            lambda: _mlp_fields(directory, seed, mlp_steps),
        ),
        "gpt": (
            "attention over the recent context",
            _run_gpt,
            lambda: _gpt_fields(directory, seed, gpt_steps),
        ),
    }

    fresh = tuple(
        _training_command(
            kind.replace("-", " "),
            description,
            run,
            directory,
            seed,
            store,
            clock,
            fields(),
        )
        for kind, (description, run, fields) in retrain_kinds.items()
    )

    def menu():
        loads = ()
        retraining = ()
        if store is not None:
            loads = tuple(
                Command(
                    f"load {name}",
                    _load_description(store.record(name)),
                    partial(load_checkpoint_session, store, seed, name),
                )
                for name in store.names()
            )
            retraining = tuple(
                _training_command(
                    f"retrain {name}",
                    description,
                    run,
                    directory,
                    seed,
                    store,
                    clock,
                    fields(),
                    continues=True,
                    pin={START_FROM: name},
                )
                for name, kind in _retrainable(store)
                for run, fields, description in (
                    (
                        retrain_kinds[kind][1],
                        retrain_kinds[kind][2],
                        retrain_kinds[kind][0],
                    ),
                )
            )
        return {
            "new": (counted, *fresh),
            "load": loads,
            "retrain": retraining,
            "sessions": _sessions_group(stories, store, seed, session)
            if stories is not None
            else (),
        }

    return menu


def _retrainable(store):
    return tuple(
        (name, store.load(name).kind)
        for name in store.names()
        if store.load(name).kind in RETRAINABLE
    )
