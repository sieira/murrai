import threading
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
from murrai.application.estimates import (
    GPT_COST,
    MLP_COST,
    NEURAL_COST,
    Calibrator,
    StepClock,
    estimate_perplexity,
    estimate_seconds,
    format_seconds,
    uniform_perplexity,
)
from murrai.application.runs import (
    chip_profile,
    suggested_defaults,
    train_bigram,
    train_mlp_lm,
    train_neural_bigram,
    train_transformer_lm,
    training_device,
)
from murrai.presentation.chat.form import FRESH, START_FROM, Field
from murrai.presentation.chat.panel import Command, FormCommand
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
        name = f"{model.kind}-{Path(directory).name}"
        store.save(name, snapshot_of(model))
        kept = CheckpointRecord(
            settings=RunSettings(
                saved_at=_saved_at(clock),
                corpus=str(directory),
                seed=seed,
                corpus_chars=len(text) if cap is not None else None,
                model=model.kind,
            ),
            gradient=GradientRun(**settings) if settings else None,
        )
        store.save_record(name, kept)

    started = clock()
    try:
        result = run(text, seed, store, values, bus=bus, keeper=keep)
        bus.flush()
    finally:
        if on_event is not None:
            bus.unsubscribe(on_event)
    seconds = clock() - started

    if kept is not None:
        final_loss, perplexity = _result_info(result)
        if kept.gradient is not None and final_loss is not None:
            kept = replace(
                kept,
                gradient=replace(
                    kept.gradient,
                    final_loss=final_loss,
                    seconds=seconds,
                ),
            )
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


def _record_settings(values, shape_names):
    settings = _settings(values)
    fresh = values.get(START_FROM, FRESH) == FRESH
    settings["shape"] = (
        {name: values[name] for name in shape_names if name in values} if fresh else {}
    )
    return settings


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
        keeper=_keeping(keeper, _record_settings(values, ("merges",))),
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
        keeper=_keeping(
            keeper,
            _record_settings(values, ("window", "dimension", "hidden", "merges")),
        ),
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
        keeper=_keeping(
            keeper,
            _record_settings(
                values,
                ("depth", "dimension", "heads", "max_time", "merges"),
            ),
        ),
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


def _holdout_field():
    return Field(
        "holdout",
        "holdout",
        "float",
        0.1,
        check="from zero below one",
        about=(
            "a slice of the corpus kept out of training; perplexity is measured there"
        ),
    )


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
        about="which corpus folder trains the model",
    )


def _size_field():
    return Field(
        "size",
        "size",
        "choice",
        "all",
        choices=("all", "100k", "500k"),
        about=(
            "how much of the corpus to train on; the small sizes are for quick checks"
        ),
    )


def _seed_field(seed):
    return Field(
        "seed",
        "seed",
        "int",
        seed,
        check="whole number",
        about="the random start; same seed, same run",
    )


def _start_field(store):
    names = store.names() if store is not None else ()
    return Field(
        START_FROM,
        "start from",
        "choice",
        FRESH,
        choices=(FRESH, *names),
        about=(
            "continue from a saved checkpoint instead of fresh;"
            " the shape fields lock to what it was trained with"
        ),
    )


def _device_field(default):
    return Field(
        "device",
        "device",
        "choice",
        default,
        choices=("auto", "prefer gpu", "cpu"),
        about="which chip trains the model; auto picks the gpu when one is present",
    )


def _merges_field():
    return Field(
        "merges",
        "merges",
        "int",
        300,
        check="whole above zero",
        shape=True,
        about=(
            "byte pairs the tokenizer learns from the corpus; more merges means"
            " longer tokens, so the same context length covers more characters"
        ),
    )


def _counted_fields(directory):
    return (
        _text_field(directory),
        _size_field(),
        _holdout_field(),
        _merges_field(),
    )


def _steps_field(steps, about):
    return Field("steps", "steps", "int", steps, check="whole above zero", about=about)


def _batch_field(chip, fixed, about):
    return Field(
        "batch",
        "batch",
        "int",
        _tuned(chip, "batch", fixed),
        check="whole above zero",
        about=about,
    )


def _learning_rate_field(chip, fixed, about):
    return Field(
        "learning_rate",
        "learning rate",
        "float",
        fixed,
        check="number above zero",
        about=about,
    )


def _neural_fields(directory, seed, store, steps, chip):
    return (
        _text_field(directory),
        _size_field(),
        _steps_field(
            steps,
            "how many batches to train on; more steps means more passes"
            " over the corpus, until improvements stop",
        ),
        _batch_field(
            chip,
            64,
            "how many sequences train at once; a bigger batch averages"
            " more text into each step",
        ),
        _learning_rate_field(
            chip,
            3.0,
            "the step size for gradient descent; small learns slowly,"
            " large overshoots and the loss jitters",
        ),
        Field(
            "final_learning_rate",
            "final learning rate",
            "float",
            0.1,
            check="number above zero",
            about="where the step size ends; the last steps land gently",
        ),
        _seed_field(seed),
        _merges_field(),
        _start_field(store),
    )


def _mlp_fields(directory, seed, store, steps, chip):
    return (
        _text_field(directory),
        _size_field(),
        _steps_field(
            steps,
            "how many batches to train on; more steps means more passes"
            " over the corpus, until improvements stop",
        ),
        _batch_field(
            chip,
            256,
            "how many sequences train at once; a bigger batch averages"
            " more text into each step",
        ),
        _learning_rate_field(
            chip,
            0.5,
            "the step size for gradient descent; small learns slowly,"
            " large overshoots and the loss jitters",
        ),
        Field(
            "window",
            "window",
            "int",
            8,
            check="whole above zero",
            shape=True,
            about="how many previous tokens the model sees at once, its context window",
        ),
        Field(
            "dimension",
            "dimension",
            "int",
            _tuned(chip, "dimension", 32),
            check="whole above zero",
            shape=True,
            about=(
                "the width of the model's vectors; the main size knob;"
                " too small and extra steps buy nothing"
            ),
        ),
        Field(
            "hidden",
            "hidden",
            "int",
            _tuned(chip, "hidden", 256),
            check="whole above zero",
            shape=True,
            about="the size of the hidden layer between the lookups and the prediction",
        ),
        Field(
            "patience",
            "patience",
            "int",
            1000,
            check="whole zero or more",
            about="how many steps without improvement before the learning rate shrinks",
        ),
        Field(
            "shrink",
            "shrink",
            "float",
            0.5,
            check="between zero and one",
            about="how much the learning rate shrinks each time patience runs out",
        ),
        _holdout_field(),
        _seed_field(seed),
        _device_field("auto"),
        _merges_field(),
        _start_field(store),
    )


def _gpt_fields(directory, seed, store, steps, chip):
    return (
        _text_field(directory),
        _size_field(),
        _steps_field(
            steps,
            "how many batches to train on; more steps means more passes"
            " over the corpus, until improvements stop",
        ),
        _batch_field(
            chip,
            256,
            "how many sequences train at once; a bigger batch averages"
            " more text into each step",
        ),
        _learning_rate_field(
            chip,
            0.001,
            "the step size for gradient descent; small learns slowly,"
            " large overshoots and the loss jitters",
        ),
        Field(
            "depth",
            "depth",
            "int",
            _tuned(chip, "depth", 4),
            check="whole above zero",
            shape=True,
            about=(
                "how many attention blocks are stacked;"
                " each block re-reads the same window"
            ),
        ),
        Field(
            "dimension",
            "dimension",
            "int",
            _tuned(chip, "dimension", 64),
            check="whole above zero",
            shape=True,
            about=(
                "the width of the model's vectors; the main size knob;"
                " too small and extra steps buy nothing"
            ),
        ),
        Field(
            "heads",
            "heads",
            "int",
            1,
            check="whole above zero",
            shape=True,
            about=(
                "how many slices attention is split into; at a small dimension"
                " each head gets narrow, so fewer is often better"
            ),
        ),
        Field(
            "max_time",
            "context length",
            "int",
            _tuned(chip, "max_time", 64),
            check="whole above zero",
            shape=True,
            about="how many previous tokens attention can read at once",
        ),
        Field(
            "patience",
            "patience",
            "int",
            200,
            check="whole zero or more",
            about="how many steps without improvement before the learning rate shrinks",
        ),
        Field(
            "shrink",
            "shrink",
            "float",
            0.5,
            check="between zero and one",
            about="how much the learning rate shrinks each time patience runs out",
        ),
        _holdout_field(),
        _seed_field(seed),
        _device_field("auto"),
        _merges_field(),
        _start_field(store),
    )


def _tuned(chip, name, fixed):
    return suggested_defaults(chip).get(name, fixed)


def _records(store):
    if store is None:
        return []
    return [store.record(name) for name in store.names()]


def _cost_config(store, values, shape_names, shapes):
    name = values.get(START_FROM, FRESH)
    if name != FRESH and store is not None:
        return _continue_config(store, name, values, shape_names, shapes)
    if any(field not in values for field in shape_names):
        return None
    config = {field: values[field] for field in shape_names}
    config["batch"] = values["batch"]
    config["steps"] = values["steps"]
    return config


def _continue_config(store, name, values, shape_names, shapes):
    if name not in shapes:
        try:
            snapshot = store.load(name)
        except Exception:
            # the estimate renders on every keystroke; a broken
            # checkpoint must not take the chat down
            return None
        shape = {
            key: snapshot.state[key]
            for key in shape_names
            if key != "merges" and key in snapshot.state
        }
        shape["merges"] = len(snapshot.merges)
        shapes[name] = shape
    config = dict(shapes[name])
    config["batch"] = values["batch"]
    config["steps"] = values["steps"]
    return config


def _measure(run, directory, seed, fields, shape_names):
    defaults = {field.name: field.default for field in fields}

    def measure():
        burst = {**defaults, "steps": 10, "size": "100k"}
        clock = StepClock()
        run(_sliced_text(directory, burst), seed, None, burst, bus=clock)
        config = {name: defaults[name] for name in shape_names}
        config.update(batch=defaults["batch"], merges=defaults["merges"])
        return clock.per_step(), config

    return measure


def _calibrate(calibrator):
    def calibrate():
        calibrator.begin()
        threading.Thread(target=calibrator.refresh, daemon=True).start()

    return calibrate


def _estimates(kind, store, directory, calibrator, cost, shape_names):
    shapes = {}

    def estimates(values):
        if not {"steps", "batch", "merges"} <= values.keys():
            return []
        records = _records(store)
        lines = []
        config = _cost_config(store, values, shape_names, shapes)
        if config is not None:
            projected = calibrator.project(config)
            if projected is not None:
                lines.append(
                    f"this run should take about {format_seconds(projected)}",
                )
            elif calibrator.pending:
                lines.append("measuring this chip…")
            else:
                history = estimate_seconds(records, config, kind, cost)
                if history is not None:
                    plural = "" if history.runs == 1 else "s"
                    lines.append(
                        f"{history.text} (from {history.runs} past run{plural})",
                    )
        evidence = estimate_perplexity(
            records,
            kind,
            str(_chosen_directory(directory, values)),
        )
        if evidence is not None:
            lines.append(evidence.text)
        else:
            floor = uniform_perplexity(values["merges"])
            lines.append(
                f"a uniform guess over the {floor} pieces lands at ppl {floor}",
            )
        return lines

    return estimates


def _training_command(
    name,
    description,
    run,
    directory,
    seed,
    store,
    clock,
    fields,
    kind=None,
    cost=None,
    calibrators=None,
):
    shape_names = tuple(field.name for field in fields if field.shape)
    estimates = calibrate = None
    if cost is not None:
        calibrator = calibrators.get(kind)
        if calibrator is None:
            calibrator = Calibrator(
                _measure(run, directory, seed, fields, shape_names),
                cost,
            )
            calibrators[kind] = calibrator
        estimates = _estimates(kind, store, directory, calibrator, cost, shape_names)
        calibrate = _calibrate(calibrator)

    def bake(values):
        return Command(
            name,
            description,
            partial(_trained_chat_session, run, directory, seed, store, clock, values),
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
        estimates=estimates,
        calibrate=calibrate,
    )


def load_checkpoint_session(store, seed=0, name=None):
    if name is None:
        name = store.names()[-1]
    model = restore(store.load(name))
    return ChatSession(model, seed, checkpoint=name)


def chat_commands(
    directory,
    seed=0,
    store=None,
    neural_steps=2000,
    mlp_steps=10000,
    gpt_steps=2000,
    chip=None,
    calibrators=None,
    clock=time.time,
):
    chip = chip_profile() if chip is None else chip
    calibrators = {} if calibrators is None else calibrators
    commands = []
    if store is not None:
        for name in store.names():
            commands.append(
                Command(
                    f"load {name}",
                    _load_description(store.record(name)),
                    lambda name=name: load_checkpoint_session(store, seed, name),
                ),
            )
    commands.extend(
        [
            _training_command(
                "counted bigram",
                "next-char odds tallied straight from the corpus",
                _run_counted,
                directory,
                seed,
                store,
                clock,
                _counted_fields(directory),
            ),
            _training_command(
                "retrain neural bigram",
                "same memory, trained by gradient descent",
                _run_neural,
                directory,
                seed,
                store,
                clock,
                _neural_fields(directory, seed, store, neural_steps, chip),
                kind="neural-bigram",
                cost=NEURAL_COST,
                calibrators=calibrators,
            ),
            _training_command(
                "retrain windowed mlp",
                "eight tokens of memory through a hidden layer",
                _run_mlp,
                directory,
                seed,
                store,
                clock,
                _mlp_fields(directory, seed, store, mlp_steps, chip),
                kind="windowed-mlp",
                cost=MLP_COST,
                calibrators=calibrators,
            ),
            _training_command(
                "retrain gpt",
                "attention over the last sixty-four tokens",
                _run_gpt,
                directory,
                seed,
                store,
                clock,
                _gpt_fields(directory, seed, store, gpt_steps, chip),
                kind="gpt",
                cost=GPT_COST,
                calibrators=calibrators,
            ),
        ],
    )
    return commands
