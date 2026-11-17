import pickle
import time
from types import SimpleNamespace

import pytest
import torch

import murrai.composition.chat as murrai_composition_chat
from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from murrai.application.checkpoints import (
    CheckpointRecord,
    GptShape,
    GradientRun,
    RunSettings,
    Snapshot,
)
from murrai.application.ports import TrainingEvent
from murrai.application.runs import training_device
from murrai.composition.chat import chat_commands
from murrai.domain.transformer import TransformerLM
from murrai.presentation.chat.form import TrainingForm
from murrai.presentation.chat.panel import FormCommand


@pytest.fixture
def corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "story.txt").write_text("aabbaabb" * 20)
    return corpus


def test_chat_commands_list_the_models(corpus):
    commands = chat_commands(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
    )
    assert [command.name for command in commands] == [
        "counted bigram",
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain gpt",
    ]
    counted, neural, mlp, gpt = commands
    assert counted.trains
    assert neural.trains
    assert mlp.trains
    assert gpt.trains


def _command(corpus, name, **kwargs):
    return next(c for c in chat_commands(corpus, **kwargs) if c.name == name)


def test_counted_command_builds_a_working_session(corpus):
    session = _command(corpus, "counted bigram", seed=0).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_neural_command_builds_a_working_session(corpus):
    session = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_command_builds_a_working_session(corpus):
    session = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_gpt_command_builds_a_working_session(corpus):
    session = _command(corpus, "retrain gpt", seed=0, gpt_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_session_story_carries_generated_text(corpus):
    session = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20).build()
    session.extend_user("a")
    session.extend_generated("b")
    session.extend_user("a")
    assert session.text == "aba"
    assert session.segments() == [
        ("a", "entered"),
        ("b", "generated"),
        ("a", "entered"),
    ]


def test_no_store_means_no_load_commands(corpus):
    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert not any(command.name.startswith("load ") for command in commands)


def test_every_field_of_every_training_form_carries_an_explanation(corpus):
    commands = chat_commands(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
    )
    for command in commands:
        if not isinstance(command, FormCommand):
            continue
        for field in command.fields:
            assert field.about, f"{command.name} · {field.name} has no explanation"


def test_the_gpt_defaults_follow_the_chip(corpus):
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, chip="mps")
    defaults = {field.name: field.default for field in gpt.fields}

    assert defaults["batch"] == 128
    assert defaults["dimension"] == 128
    assert defaults["depth"] == 6
    assert defaults["max_time"] == 128
    assert defaults["heads"] == 1


def test_the_mlp_defaults_follow_the_chip(corpus):
    mlp = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20, chip="cpu")
    defaults = {field.name: field.default for field in mlp.fields}

    assert defaults["batch"] == 64
    assert defaults["dimension"] == 64
    assert defaults["hidden"] == 128
    assert defaults["window"] == 8


def test_the_chip_never_touches_the_steps_or_the_seed(corpus):
    gpt = _command(corpus, "retrain gpt", seed=7, gpt_steps=99, chip="mps")
    defaults = {field.name: field.default for field in gpt.fields}

    assert defaults["steps"] == 99
    assert defaults["seed"] == 7


def test_a_chip_left_unpicked_reports_this_machine(corpus, monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20)
    defaults = {field.name: field.default for field in gpt.fields}

    assert defaults["dimension"] == 128


def test_a_fresh_run_records_its_shape_model_and_duration(corpus):
    store = InMemoryCheckpointStore()
    stamps = iter([1_800_000_000, 1_800_000_000, 1_800_006_000])
    _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        store=store,
        clock=lambda: next(stamps),
        chip="cpu",
    ).build()

    record = store.record("gpt-corpus")
    assert record.settings.model == "gpt"
    assert record.gradient.shape == GptShape(
        merges=300,
        depth=4,
        dimension=64,
        heads=1,
        max_time=64,
    )
    assert record.gradient.seconds == pytest.approx(6000)


def test_a_continue_run_records_no_shape(corpus):
    store = InMemoryCheckpointStore()
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store, chip="cpu")
    gpt.build()
    gpt.bake({**_defaults(gpt), "start_from": "gpt-corpus"}).build()

    assert store.record("gpt-corpus").gradient.shape is None


def test_the_gpt_form_estimates_from_history(corpus, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no chip")

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", boom)
    store = InMemoryCheckpointStore()
    store.save("gpt-corpus", Snapshot(kind="gpt", merges=(), state={}))
    store.save_record(
        "gpt-corpus",
        CheckpointRecord(
            settings=RunSettings(
                saved_at="2026-11-16T10:16:59+00:00",
                corpus=str(corpus),
                seed=0,
                model="gpt",
            ),
            gradient=GradientRun(
                steps=20,
                batch=64,
                learning_rate=0.001,
                seconds=600.0,
                shape=GptShape(
                    merges=300,
                    depth=4,
                    dimension=64,
                    heads=1,
                    max_time=64,
                ),
            ),
            held_out_perplexity=11.7,
        ),
    )
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store, chip="cpu")

    lines = []
    for _ in range(200):
        lines = gpt.estimates(_defaults(gpt))
        if lines:
            break
        time.sleep(0.01)

    assert lines == ["ETA: about 10 min"]


def test_the_form_stays_silent_when_the_rehearsal_fails_without_history(
    corpus,
    monkeypatch,
):
    def boom(*args, **kwargs):
        raise RuntimeError("no chip")

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", boom)
    calibrators = {}
    gpt = _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        chip="cpu",
        calibrators=calibrators,
    )
    defaults = _defaults(gpt)

    gpt.estimates(defaults)
    calibrator = next(iter(calibrators.values()))
    for _ in range(200):
        if not calibrator.pending:
            break
        time.sleep(0.01)

    assert gpt.estimates(defaults) == []


def test_reading_the_estimates_starts_the_rehearsal(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_transformer_lm")
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, chip="cpu")

    gpt.estimates(_defaults(gpt))
    for _ in range(200):
        if "kwargs" in captured:
            break
        time.sleep(0.01)

    assert captured["kwargs"]["steps"] == 10


def test_a_device_change_measures_the_new_chip(corpus, monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    real = murrai_composition_chat.train_transformer_lm
    rehearsals = []

    def counting(text, **kwargs):
        rehearsals.append(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", counting)
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, chip="cpu")
    defaults = _defaults(gpt)

    for _ in range(200):
        if gpt.estimates(defaults):
            break
        time.sleep(0.01)
    assert len(rehearsals) == 1

    values = {**defaults, "device": "cpu"}
    for _ in range(200):
        if gpt.estimates(values):
            break
        time.sleep(0.01)

    assert len(rehearsals) == 2
    assert str(rehearsals[1]["device"]) == "cpu"


def test_auto_and_prefer_gpu_share_one_measurement(corpus, monkeypatch):
    real = murrai_composition_chat.train_transformer_lm
    rehearsals = []

    def counting(text, **kwargs):
        rehearsals.append(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", counting)
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, chip="cpu")
    defaults = _defaults(gpt)

    for _ in range(200):
        if gpt.estimates(defaults):
            break
        time.sleep(0.01)
    assert len(rehearsals) == 1

    gpu = {**defaults, "device": "prefer gpu"}
    assert gpt.estimates(gpu)
    assert len(rehearsals) == 1


def test_the_form_waits_for_the_rehearsal_before_starting_a_run(
    corpus,
    monkeypatch,
):
    real = murrai_composition_chat.train_transformer_lm

    def slow(text, **kwargs):
        time.sleep(0.3)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", slow)
    calibrators = {}
    gpt = _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        chip="cpu",
        calibrators=calibrators,
    )
    defaults = _defaults(gpt)

    gpt.estimates(defaults)
    calibrator = next(iter(calibrators.values()))
    assert calibrator.pending

    gpt.bake(defaults)

    assert not calibrator.pending


def test_a_vanished_checkpoint_is_ignored_by_the_estimate(corpus):
    store = InMemoryCheckpointStore()
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store, chip="cpu")
    values = {**_defaults(gpt), "start_from": "gone"}

    assert gpt.estimates(values) == []


def test_the_form_shows_the_eta_when_continuing_from_a_checkpoint(
    corpus,
    monkeypatch,
):
    def boom(*args, **kwargs):
        raise RuntimeError("no chip")

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", boom)
    store = InMemoryCheckpointStore()
    store.save(
        "gpt-corpus",
        Snapshot(
            kind="gpt",
            merges=(),
            state={"depth": 4, "dimension": 64, "heads": 1, "max_time": 64},
        ),
    )
    store.save_record(
        "gpt-corpus",
        CheckpointRecord(
            settings=RunSettings(
                saved_at="2026-11-16T10:16:59+00:00",
                corpus=str(corpus),
                seed=0,
                model="gpt",
            ),
            gradient=GradientRun(
                steps=20,
                batch=64,
                learning_rate=0.001,
                seconds=600.0,
                shape=GptShape(
                    merges=300,
                    depth=4,
                    dimension=64,
                    heads=1,
                    max_time=64,
                ),
            ),
        ),
    )
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store, chip="cpu")
    form = TrainingForm(gpt)
    form.move_choice("start_from", 1)
    values = form.snapshot_values()

    assert "merges" not in values
    lines = []
    for _ in range(200):
        lines = gpt.estimates(values)
        if lines:
            break
        time.sleep(0.01)

    assert lines == ["ETA: about 10 min"]


def test_a_failed_checkpoint_load_is_read_once(corpus, monkeypatch):
    store = InMemoryCheckpointStore()
    loads = []

    def failing_load(name):
        loads.append(name)
        raise RuntimeError("weights are gone")

    monkeypatch.setattr(store, "load", failing_load)
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store, chip="cpu")
    values = {**_defaults(gpt), "start_from": "gone"}

    gpt.estimates(values)
    gpt.estimates(values)
    gpt.estimates(values)

    assert len(loads) == 1


def test_a_corrupt_record_file_is_ignored(corpus, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no chip")

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", boom)
    store = FileCheckpointStore(tmp_path)
    store.save("gpt-corpus", Snapshot(kind="gpt", merges=(), state={}))
    (tmp_path / "gpt-corpus.json").write_text("{truncated")
    calibrators = {}
    gpt = _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        store=store,
        chip="cpu",
        calibrators=calibrators,
    )
    defaults = _defaults(gpt)

    gpt.estimates(defaults)
    calibrator = next(iter(calibrators.values()))
    for _ in range(200):
        if not calibrator.pending:
            break
        time.sleep(0.01)

    assert gpt.estimates(defaults) == []


def test_one_calibration_serves_rebuilt_commands(corpus, monkeypatch):
    real = murrai_composition_chat.train_transformer_lm
    rehearsals = []

    def counting(text, **kwargs):
        rehearsals.append(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", counting)
    calibrators = {}
    first = next(
        c
        for c in chat_commands(
            corpus,
            seed=0,
            gpt_steps=20,
            chip="cpu",
            calibrators=calibrators,
        )
        if c.name == "retrain gpt"
    )
    first.estimates(_defaults(first))
    for _ in range(200):
        if first.estimates(_defaults(first)):
            break
        time.sleep(0.01)
    assert len(rehearsals) == 1

    second = next(
        c
        for c in chat_commands(
            corpus,
            seed=0,
            gpt_steps=20,
            chip="cpu",
            calibrators=calibrators,
        )
        if c.name == "retrain gpt"
    )
    second.estimates(_defaults(second))
    for _ in range(200):
        if second.estimates(_defaults(second)):
            break
        time.sleep(0.01)

    assert len(rehearsals) == 1


def test_training_commands_save_a_checkpoint(corpus):
    store = InMemoryCheckpointStore()
    chat_commands(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )
    _command(corpus, "counted bigram", store=store).build()
    assert store.names() == ["counted-bigram-corpus"]
    _command(corpus, "retrain neural bigram", store=store, neural_steps=20).build()
    assert store.names() == ["counted-bigram-corpus", "neural-bigram-corpus"]
    _command(corpus, "retrain windowed mlp", store=store, mlp_steps=20).build()
    assert store.names() == [
        "counted-bigram-corpus",
        "neural-bigram-corpus",
        "windowed-mlp-corpus",
    ]
    _command(corpus, "retrain gpt", store=store, gpt_steps=20).build()
    assert store.names() == [
        "counted-bigram-corpus",
        "gpt-corpus",
        "neural-bigram-corpus",
        "windowed-mlp-corpus",
    ]


def _clock_at_2027():
    return 1_800_000_000


def test_the_neural_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
        clock=_clock_at_2027,
        chip="cpu",
    ).build()

    record = store.record("neural-bigram-corpus")
    assert record.settings.saved_at == "2027-01-15T08:00:00+00:00"
    assert record.settings.corpus == str(corpus)
    assert record.settings.seed == 0
    assert record.gradient.steps == 20
    assert record.gradient.batch == 64
    assert record.gradient.learning_rate == 3.0
    assert isinstance(record.gradient.final_loss, float)
    assert record.held_out_perplexity is None


def test_the_counted_checkpoint_records_no_steps(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "counted bigram",
        seed=0,
        store=store,
        clock=_clock_at_2027,
    ).build()

    record = store.record("counted-bigram-corpus")
    assert record.settings.saved_at == "2027-01-15T08:00:00+00:00"
    assert record.gradient is None
    assert isinstance(record.held_out_perplexity, float)


def test_the_mlp_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain windowed mlp",
        seed=0,
        mlp_steps=20,
        store=store,
        clock=_clock_at_2027,
        chip="cpu",
    ).build()

    record = store.record("windowed-mlp-corpus")
    assert record.gradient.learning_rate == 0.5
    assert record.gradient.steps == 20
    assert record.gradient.batch == 64
    assert isinstance(record.held_out_perplexity, float)


def test_the_gpt_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        store=store,
        clock=_clock_at_2027,
    ).build()

    assert store.record("gpt-corpus").gradient.learning_rate == 0.001


def test_the_checkpoint_records_the_last_loss_of_the_run(corpus, monkeypatch):
    real = murrai_composition_chat.train_neural_bigram
    captured = {}

    def recorder(text, **kwargs):
        result = real(text, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(murrai_composition_chat, "train_neural_bigram", recorder)
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    ).build()

    record = store.record("neural-bigram-corpus")
    assert record.gradient.final_loss == captured["result"].loss_curve[-1]


def test_the_load_command_describes_the_checkpoint(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
        clock=_clock_at_2027,
    ).build()
    final_loss = store.record("neural-bigram-corpus").gradient.final_loss

    loader = _command(
        corpus,
        "load neural-bigram-corpus",
        seed=0,
        store=store,
        clock=_clock_at_2027,
    )
    assert loader.description == (
        f"saved 2027-01-15 08:00 · 20 steps · loss {final_loss:.2f}"
    )


def test_a_checkpoint_without_a_record_keeps_the_old_load_line(corpus):
    store = InMemoryCheckpointStore()
    store.save(
        "neural-bigram-corpus",
        Snapshot(kind="neural-bigram", merges=(), state={}),
    )
    loader = _command(corpus, "load neural-bigram-corpus", seed=0, store=store)
    assert loader.description == "pick up this trained model"


def test_the_load_command_restores_the_trained_model(corpus):
    store = InMemoryCheckpointStore()
    trainer = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    trainer.build()

    loader = _command(corpus, "load neural-bigram-corpus", seed=0, store=store)
    assert not loader.trains
    session = loader.build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_training_commands_stream_events_to_the_listener(corpus):
    events = []

    command = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
    )
    command.build(on_event=events.append)

    training = [event for event in events if event.phase == "training"]
    assert len(training) == 20
    assert all(isinstance(event, TrainingEvent) for event in events)


def test_commands_offer_loading_and_retraining(corpus):
    store = InMemoryCheckpointStore()
    trainer = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    trainer.build()

    commands = chat_commands(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )

    names = [command.name for command in commands]
    assert "load neural-bigram-corpus" in names
    assert "counted bigram" in names
    assert "retrain neural bigram" in names
    assert "retrain windowed mlp" in names
    assert "retrain gpt" in names


def test_the_load_command_builds_a_working_session(corpus):
    store = InMemoryCheckpointStore()
    trainer = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    trainer.build()

    loader = _command(corpus, "load neural-bigram-corpus", seed=0, store=store)
    session = loader.build()

    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_training_saves_the_model_before_the_metrics_run(tmp_path, monkeypatch):
    def broken_perplexity(self, text):
        raise RuntimeError("metrics blew up")

    monkeypatch.setattr(TransformerLM, "perplexity", broken_perplexity)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "story.txt").write_text("aabbaabb" * 20)
    store = FileCheckpointStore(tmp_path / "checkpoints")
    command = _command(corpus, "retrain gpt", seed=0, gpt_steps=5, store=store)
    with pytest.raises(RuntimeError):
        command.build()
    assert store.names() == ["gpt-corpus"]
    gpt_record = store.record("gpt-corpus")
    assert gpt_record.gradient.steps == 5
    assert gpt_record.gradient.final_loss is None


def test_the_retrain_command_trains_and_saves(corpus):
    store = InMemoryCheckpointStore()

    retrainer = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    session = retrainer.build()

    assert store.names() == ["neural-bigram-corpus"]
    session.extend_user("a")
    assert session.rollout("a", 5)


def test_no_store_means_counted_and_retraining_commands(corpus):
    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert [command.name for command in commands] == [
        "counted bigram",
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain gpt",
    ]


def test_training_commands_survive_a_trip_to_another_process(corpus):
    import pickle

    commands = chat_commands(corpus, seed=0, neural_steps=20)
    trains = [c for c in commands if c.trains]
    assert [c.name for c in trains] == [
        "counted bigram",
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain gpt",
    ]

    for command in trains:
        baked = command.bake({f.name: f.default for f in command.fields})
        assert pickle.loads(pickle.dumps(baked)).name == command.name

    revived = pickle.loads(
        pickle.dumps(trains[0].bake({f.name: f.default for f in trains[0].fields})),
    )
    assert revived.build().rollout("a", 5)


def test_the_gpt_command_trains_on_the_policy_device(corpus, monkeypatch):
    real = murrai_composition_chat.train_transformer_lm
    captured = {}

    def recorder(text, **kwargs):
        captured.update(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", recorder)
    _command(corpus, "retrain gpt", seed=0, gpt_steps=20).build()
    assert captured["device"] == training_device(prefer_mps=True)


def test_the_mlp_command_trains_on_the_policy_device(corpus, monkeypatch):
    real = murrai_composition_chat.train_mlp_lm
    captured = {}

    def recorder(text, **kwargs):
        captured.update(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_mlp_lm", recorder)
    _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20).build()
    assert captured["device"] == training_device(prefer_mps=False)


def _defaults(command):
    return {field.name: field.default for field in command.fields}


def _field(corpus, command_name, field_name, **kwargs):
    command = _command(corpus, command_name, **kwargs)
    return next(field for field in command.fields if field.name == field_name)


def _available(monkeypatch, cuda, mps):
    def fake(available):
        return type("_", (), {"is_available": staticmethod(lambda: available)})

    monkeypatch.setattr(torch, "cuda", fake(cuda))
    monkeypatch.setattr(torch.backends, "mps", fake(mps))


def _stub_trainer(monkeypatch, name, captured):
    def stub(text, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=SimpleNamespace(kind="stub"), loss_curve=[1.0])

    monkeypatch.setattr(murrai_composition_chat, name, stub)


def _capturing(monkeypatch, name, **capture):
    real = getattr(murrai_composition_chat, name)

    def recorder(*args, **kwargs):
        capture["kwargs"] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, name, recorder)
    return capture


def test_the_counted_command_offers_its_knobs(corpus):
    counted = _command(corpus, "counted bigram", seed=0)

    assert [(f.name, f.label, f.kind, f.default) for f in counted.fields] == [
        ("text", "text", "choice", corpus.name),
        ("size", "size", "choice", "all"),
        ("holdout", "holdout", "float", 0.1),
        ("merges", "merges", "int", 300),
    ]
    assert counted.fields[-1].check == "whole above zero"


def test_the_neural_command_offers_its_knobs(corpus):
    fields = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        chip="cpu",
    ).fields

    assert [(f.name, f.label, f.kind, f.default) for f in fields] == [
        ("text", "text", "choice", corpus.name),
        ("size", "size", "choice", "all"),
        ("steps", "steps", "int", 20),
        ("batch", "batch", "int", 64),
        ("learning_rate", "learning rate", "float", 3.0),
        ("final_learning_rate", "final learning rate", "float", 0.1),
        ("seed", "seed", "int", 0),
        ("merges", "merges", "int", 300),
        ("start_from", "start from", "choice", "fresh"),
    ]
    assert fields[-1].choices == ("fresh",)


def test_the_start_from_choice_lists_the_saved_models(corpus):
    store = InMemoryCheckpointStore()
    store.save("gpt-corpus", Snapshot(kind="gpt", merges=(), state={}))
    store.save(
        "neural-bigram-corpus",
        Snapshot(kind="neural-bigram", merges=(), state={}),
    )

    field = _field(
        corpus,
        "retrain neural bigram",
        "start_from",
        seed=0,
        neural_steps=20,
        store=store,
    )

    assert field.choices == ("fresh", "gpt-corpus", "neural-bigram-corpus")


def test_the_mlp_command_offers_its_knobs(corpus):
    fields = _command(
        corpus,
        "retrain windowed mlp",
        seed=0,
        mlp_steps=20,
        chip="cpu",
    ).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
        ("text", "text", corpus.name),
        ("size", "size", "all"),
        ("steps", "steps", 20),
        ("batch", "batch", 64),
        ("learning_rate", "learning rate", 0.5),
        ("window", "window", 8),
        ("dimension", "dimension", 64),
        ("hidden", "hidden", 128),
        ("patience", "patience", 1000),
        ("shrink", "shrink", 0.5),
        ("holdout", "holdout", 0.1),
        ("seed", "seed", 0),
        ("device", "device", "auto"),
        ("merges", "merges", 300),
        ("start_from", "start from", "fresh"),
    ]
    assert [f.name for f in fields if f.shape] == [
        "window",
        "dimension",
        "hidden",
        "merges",
    ]
    assert fields[-3].choices == ("auto", "prefer gpu", "cpu")
    assert fields[-1].choices == ("fresh",)


def test_the_gpt_command_offers_its_knobs(corpus):
    fields = _command(
        corpus,
        "retrain gpt",
        seed=0,
        gpt_steps=20,
        chip="cpu",
    ).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
        ("text", "text", corpus.name),
        ("size", "size", "all"),
        ("steps", "steps", 20),
        ("batch", "batch", 64),
        ("learning_rate", "learning rate", 0.001),
        ("depth", "depth", 4),
        ("dimension", "dimension", 64),
        ("heads", "heads", 1),
        ("max_time", "context length", 64),
        ("patience", "patience", 200),
        ("shrink", "shrink", 0.5),
        ("holdout", "holdout", 0.1),
        ("seed", "seed", 0),
        ("device", "device", "auto"),
        ("merges", "merges", 300),
        ("start_from", "start from", "fresh"),
    ]
    assert [f.name for f in fields if f.shape] == [
        "depth",
        "dimension",
        "heads",
        "max_time",
        "merges",
    ]
    assert fields[-3].choices == ("auto", "prefer gpu", "cpu")
    assert fields[-1].choices == ("fresh",)


def test_the_baked_command_trains_with_the_given_values(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)

    command.bake(
        {**_defaults(command), "steps": 5, "batch": 7, "learning_rate": 1.5},
    ).build()

    assert captured["kwargs"]["steps"] == 5
    assert captured["kwargs"]["batch"] == 7
    assert captured["kwargs"]["learning_rate"] == 1.5
    assert captured["kwargs"]["final_learning_rate"] == 0.1
    assert captured["kwargs"]["seed"] == 0


def test_a_chosen_savegame_flows_to_the_trainer_by_its_name(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    store = InMemoryCheckpointStore()
    trainer = _command(
        corpus,
        "retrain neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    trainer.build()

    baked = trainer.bake({**_defaults(trainer), "start_from": "neural-bigram-corpus"})

    baked.build()
    assert captured["kwargs"]["checkpoint"].kind == "neural-bigram"


def test_a_fresh_start_hands_the_trainer_no_snapshot(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)

    command.bake(_defaults(command)).build()

    assert captured["kwargs"]["checkpoint"] is None


def test_a_saved_start_leaves_the_shape_to_the_savegame(corpus, monkeypatch):
    gpt_captured = _capturing(monkeypatch, "train_transformer_lm")
    mlp_captured = _capturing(monkeypatch, "train_mlp_lm")
    store = InMemoryCheckpointStore()
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20, store=store)
    mlp = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20, store=store)
    gpt.build()
    mlp.build()

    gpt.bake(
        {**_defaults(gpt), "start_from": "gpt-corpus", "depth": 9, "max_time": 128},
    ).build()
    mlp.bake(
        {
            **_defaults(mlp),
            "start_from": "windowed-mlp-corpus",
            "window": 3,
            "hidden": 9,
        },
    ).build()

    for captured in (gpt_captured, mlp_captured):
        assert captured["kwargs"]["checkpoint"].kind in ("gpt", "windowed-mlp")
        for shape in (
            "depth",
            "dimension",
            "heads",
            "max_time",
            "window",
            "hidden",
            "merges",
        ):
            assert shape not in captured["kwargs"]


def test_a_baked_command_survives_the_trip_to_another_process(corpus):
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)
    baked = command.bake({**_defaults(command), "steps": 5})

    revived = pickle.loads(pickle.dumps(baked))

    assert revived.name == baked.name
    assert revived.build().rollout("a", 5)


def test_the_device_choice_picks_where_the_training_runs(corpus, monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    mlp_captured, gpt_captured = {}, {}
    _stub_trainer(monkeypatch, "train_mlp_lm", mlp_captured)
    _stub_trainer(monkeypatch, "train_transformer_lm", gpt_captured)
    mlp = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20)
    gpt = _command(corpus, "retrain gpt", seed=0, gpt_steps=20)

    mlp.bake(_defaults(mlp)).build()
    gpt.bake(_defaults(gpt)).build()
    assert mlp_captured["device"] == torch.device("cpu")
    assert gpt_captured["device"] == torch.device("mps")

    mlp.bake({**_defaults(mlp), "device": "prefer gpu"}).build()
    assert mlp_captured["device"] == torch.device("mps")

    gpt.bake({**_defaults(gpt), "device": "cpu"}).build()
    assert gpt_captured["device"] == torch.device("cpu")


def test_the_counted_bake_takes_the_holdout(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_bigram")
    counted = _command(corpus, "counted bigram", seed=0)

    counted.bake({**_defaults(counted), "holdout": 0.5}).build()

    assert captured["kwargs"]["holdout"] == 0.5


def test_the_load_line_shows_the_capped_size(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "counted bigram",
        seed=0,
        store=store,
        clock=_clock_at_2027,
    )
    store.save(
        "counted-bigram-corpus",
        Snapshot(kind="counted-bigram", merges=(), state={}),
    )
    store.save_record(
        "counted-bigram-corpus",
        CheckpointRecord(
            settings=RunSettings(
                saved_at="2027-01-15T08:00:00+00:00",
                corpus=str(corpus),
                seed=0,
                corpus_chars=100000,
            ),
        ),
    )

    loader = _command(corpus, "load counted-bigram-corpus", seed=0, store=store)
    assert "100k chars" in loader.description
