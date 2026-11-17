import pickle
from types import SimpleNamespace

import pytest
import torch

import murrai.composition.chat as murrai_composition_chat
from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from murrai.adapters.story import InMemoryStoryStore
from murrai.application.checkpoints import (
    CheckpointRecord,
    RunSettings,
    Snapshot,
    snapshot_of,
)
from murrai.application.ports import TrainingEvent
from murrai.application.runs import training_device
from murrai.application.sessions import StorySave
from murrai.composition.chat import chat_menu
from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM
from murrai.presentation.chat import ChatSession


@pytest.fixture
def corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "story.txt").write_text("aabbaabb" * 20)
    return corpus


@pytest.fixture
def live_story():
    model = Bigram.fit("abab", TokenizerEncoderDecoder())
    session = ChatSession(model, seed=0, checkpoint="gpt-corpus")
    session.extend_user("ab")
    session.extend_generated("b")
    return session


def test_the_sessions_menu_saves_the_live_story(corpus, live_story):
    stories = InMemoryStoryStore()
    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        stories=stories,
        session=lambda: live_story,
    )()

    (save_story,) = tree["sessions"]
    assert save_story.name == "save story"
    assert save_story.fields[0].name == "name"

    command = save_story.bake({"name": "first"})
    assert command.build() is live_story
    assert stories.load("first") == StorySave(
        ids=tuple(
            live_story._model.tokenizer.encode("ab")
            + live_story._model.tokenizer.encode("b"),
        ),
        origins=("entered", "entered", "generated"),
        checkpoint="gpt-corpus",
    )


def test_the_sessions_menu_loads_a_story_back(corpus, live_story):
    stories = InMemoryStoryStore()
    stories.save("first", live_story.to_save())
    store = InMemoryCheckpointStore()
    store.save("gpt-corpus", snapshot_of(live_story._model))
    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
        stories=stories,
        session=lambda: None,
    )()

    assert [command.name for command in tree["sessions"]] == ["load first"]
    session = tree["sessions"][0].build()
    assert session.text == "abb"
    assert session.segments() == [("ab", "entered"), ("b", "generated")]
    assert session.checkpoint == "gpt-corpus"


def test_the_sessions_menu_stays_empty_without_a_story(corpus):
    stories = InMemoryStoryStore()
    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        stories=stories,
        session=lambda: None,
    )()

    assert tree["sessions"] == ()


def test_the_menu_lists_the_models(corpus):
    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
    )()
    assert [command.name for command in tree["new"]] == [
        "counted bigram",
        "neural bigram",
        "windowed mlp",
        "gpt",
    ]
    counted, neural, mlp, gpt = tree["new"]
    assert counted.trains
    assert neural.trains
    assert mlp.trains
    assert gpt.trains


def test_the_menu_descriptions_do_not_claim_the_editable_defaults(corpus):
    tree = chat_menu(corpus, seed=0, neural_steps=20, mlp_steps=20, gpt_steps=20)()
    _, _, mlp, gpt = tree["new"]
    assert mlp.description == "a short token window through a hidden layer"
    assert gpt.description == "attention over the recent context"


def test_the_menu_offers_new_load_and_retrain(corpus):
    store = InMemoryCheckpointStore()
    store.save("gpt-corpus", Snapshot(kind="gpt", merges=(), state={}))
    store.save("counted-corpus", Snapshot(kind="counted-bigram", merges=(), state={}))

    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )()

    assert [c.name for c in tree["new"]] == [
        "counted bigram",
        "neural bigram",
        "windowed mlp",
        "gpt",
    ]
    assert [c.name for c in tree["load"]] == [
        "load counted-corpus",
        "load gpt-corpus",
    ]
    assert [c.name for c in tree["retrain"]] == ["retrain gpt-corpus"]
    retrain = tree["retrain"][0]
    assert retrain.continues
    assert retrain.pin == {"start_from": "gpt-corpus"}
    assert "start_from" not in [f.name for f in retrain.fields]
    assert tree["retrain"][0].fields == tree["new"][3].fields


def test_the_retrain_menu_lists_one_item_per_gradient_checkpoint(corpus):
    store = InMemoryCheckpointStore()
    store.save("neural-corpus", Snapshot(kind="neural-bigram", merges=(), state={}))
    store.save("mlp-corpus", Snapshot(kind="windowed-mlp", merges=(), state={}))
    store.save("counted-corpus", Snapshot(kind="counted-bigram", merges=(), state={}))

    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )()

    assert [c.name for c in tree["retrain"]] == [
        "retrain mlp-corpus",
        "retrain neural-corpus",
    ]


def test_the_menu_reloads_the_checkpoints_each_time(corpus):
    store = InMemoryCheckpointStore()
    menu = chat_menu(corpus, seed=0, store=store)

    assert menu()["load"] == ()
    store.save("gpt-corpus", Snapshot(kind="gpt", merges=(), state={}))
    assert [c.name for c in menu()["load"]] == ["load gpt-corpus"]


def test_a_new_gradient_command_trains_fresh(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    tree = chat_menu(corpus, seed=0, neural_steps=20)()

    tree["new"][1].bake(_defaults(tree["new"][1])).build()

    assert captured["kwargs"]["checkpoint"] is None


def _command(corpus, name, **kwargs):
    tree = chat_menu(corpus, **kwargs)()
    return next(
        command for group in tree.values() for command in group if command.name == name
    )


def test_counted_command_builds_a_working_session(corpus):
    session = _command(corpus, "counted bigram", seed=0).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_neural_command_builds_a_working_session(corpus):
    session = _command(corpus, "neural bigram", seed=0, neural_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_command_builds_a_working_session(corpus):
    session = _command(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_gpt_command_builds_a_working_session(corpus):
    session = _command(corpus, "gpt", seed=0, gpt_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_session_story_carries_generated_text(corpus):
    session = _command(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
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
    tree = chat_menu(corpus, seed=0, neural_steps=20, mlp_steps=20)()
    assert not any(
        command.name.startswith("load ") for group in tree.values() for command in group
    )


def test_training_commands_save_a_checkpoint(corpus):
    store = InMemoryCheckpointStore()
    _command(corpus, "counted bigram", store=store).build()
    assert store.names() == ["counted-bigram-corpus"]
    _command(corpus, "neural bigram", store=store, neural_steps=20).build()
    assert store.names() == ["counted-bigram-corpus", "neural-bigram-corpus"]
    _command(corpus, "windowed mlp", store=store, mlp_steps=20).build()
    assert store.names() == [
        "counted-bigram-corpus",
        "neural-bigram-corpus",
        "windowed-mlp-corpus",
    ]
    _command(corpus, "gpt", store=store, gpt_steps=20).build()
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
        "neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
        clock=_clock_at_2027,
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
        "windowed mlp",
        seed=0,
        mlp_steps=20,
        store=store,
        clock=_clock_at_2027,
    ).build()

    record = store.record("windowed-mlp-corpus")
    assert record.gradient.learning_rate == 0.5
    assert record.gradient.steps == 20
    assert record.gradient.batch == 256
    assert isinstance(record.held_out_perplexity, float)


def test_the_gpt_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "gpt",
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
        "neural bigram",
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
        "neural bigram",
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
        "neural bigram",
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
        "neural bigram",
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
        "neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    trainer.build()

    tree = chat_menu(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )()

    names = [command.name for group in tree.values() for command in group]
    assert "load neural-bigram-corpus" in names
    assert "counted bigram" in names
    assert "retrain neural-bigram-corpus" in names


def test_the_load_command_builds_a_working_session(corpus):
    store = InMemoryCheckpointStore()
    trainer = _command(
        corpus,
        "neural bigram",
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
    command = _command(corpus, "gpt", seed=0, gpt_steps=5, store=store)
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
        "neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )
    session = retrainer.build()

    assert store.names() == ["neural-bigram-corpus"]
    session.extend_user("a")
    assert session.rollout("a", 5)


def test_no_store_means_no_retrain_commands(corpus):
    tree = chat_menu(corpus, seed=0, neural_steps=20, mlp_steps=20)()
    assert not any(
        command.name.startswith("retrain ")
        for group in tree.values()
        for command in group
    )


def test_training_commands_survive_a_trip_to_another_process(corpus):
    import pickle

    tree = chat_menu(corpus, seed=0, neural_steps=20)()
    trains = [c for c in tree["new"] if c.trains]
    assert [c.name for c in trains] == [
        "counted bigram",
        "neural bigram",
        "windowed mlp",
        "gpt",
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
    _command(corpus, "gpt", seed=0, gpt_steps=20).build()
    assert captured["device"] == training_device(prefer_mps=True)


def test_the_mlp_command_trains_on_the_policy_device(corpus, monkeypatch):
    real = murrai_composition_chat.train_mlp_lm
    captured = {}

    def recorder(text, **kwargs):
        captured.update(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_mlp_lm", recorder)
    _command(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
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
        ("name", "name", "text", f"counted-bigram-{corpus.name}"),
        ("text", "text", "choice", corpus.name),
        ("size", "size", "choice", "all"),
        ("holdout", "holdout", "float", 0.1),
        ("merges", "merges", "int", 300),
    ]
    assert counted.fields[-1].check == "whole above zero"
    assert "start_from" not in [f.name for f in counted.fields]


def test_the_neural_command_offers_its_knobs(corpus):
    fields = _command(corpus, "neural bigram", seed=0, neural_steps=20).fields

    assert [(f.name, f.label, f.kind, f.default) for f in fields] == [
        ("name", "name", "text", f"neural-bigram-{corpus.name}"),
        ("text", "text", "choice", corpus.name),
        ("size", "size", "choice", "all"),
        ("steps", "steps", "int", 20),
        ("batch", "batch", "int", 64),
        ("learning_rate", "learning rate", "float", 3.0),
        ("final_learning_rate", "final learning rate", "float", 0.1),
        ("seed", "seed", "int", 0),
        ("merges", "merges", "int", 300),
    ]


def test_the_mlp_command_offers_its_knobs(corpus):
    fields = _command(corpus, "windowed mlp", seed=0, mlp_steps=20).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
        ("name", "name", f"windowed-mlp-{corpus.name}"),
        ("text", "text", corpus.name),
        ("size", "size", "all"),
        ("steps", "steps", 20),
        ("batch", "batch", 256),
        ("learning_rate", "learning rate", 0.5),
        ("window", "window", 8),
        ("dimension", "dimension", 32),
        ("hidden", "hidden", 256),
        ("patience", "patience", 1000),
        ("shrink", "shrink", 0.5),
        ("holdout", "holdout", 0.1),
        ("seed", "seed", 0),
        ("device", "device", "auto"),
        ("merges", "merges", 300),
    ]
    assert [f.name for f in fields if f.shape] == [
        "window",
        "dimension",
        "hidden",
        "merges",
    ]
    assert fields[-2].choices == ("auto", "prefer gpu", "cpu")


def test_the_gpt_command_offers_its_knobs(corpus):
    fields = _command(corpus, "gpt", seed=0, gpt_steps=20).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
        ("name", "name", f"gpt-{corpus.name}"),
        ("text", "text", corpus.name),
        ("size", "size", "all"),
        ("steps", "steps", 20),
        ("batch", "batch", 256),
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
    ]
    assert [f.name for f in fields if f.shape] == [
        "depth",
        "dimension",
        "heads",
        "max_time",
        "merges",
    ]
    assert fields[-2].choices == ("auto", "prefer gpu", "cpu")


def test_the_baked_command_trains_with_the_given_values(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    command = _command(corpus, "neural bigram", seed=0, neural_steps=20)

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
        "neural bigram",
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
    command = _command(corpus, "neural bigram", seed=0, neural_steps=20)

    command.bake(_defaults(command)).build()

    assert captured["kwargs"]["checkpoint"] is None


def test_a_saved_start_leaves_the_shape_to_the_savegame(corpus, monkeypatch):
    gpt_captured = _capturing(monkeypatch, "train_transformer_lm")
    mlp_captured = _capturing(monkeypatch, "train_mlp_lm")
    store = InMemoryCheckpointStore()
    _command(corpus, "gpt", seed=0, gpt_steps=20, store=store).build()
    _command(corpus, "windowed mlp", seed=0, mlp_steps=20, store=store).build()
    gpt = _command(corpus, "retrain gpt-corpus", seed=0, gpt_steps=20, store=store)
    mlp = _command(
        corpus,
        "retrain windowed-mlp-corpus",
        seed=0,
        mlp_steps=20,
        store=store,
    )

    gpt.bake({**_defaults(gpt), "depth": 9, "max_time": 128}).build()
    mlp.bake({**_defaults(mlp), "window": 3, "hidden": 9}).build()

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
    command = _command(corpus, "neural bigram", seed=0, neural_steps=20)
    baked = command.bake({**_defaults(command), "steps": 5})

    revived = pickle.loads(pickle.dumps(baked))

    assert revived.name == baked.name
    assert revived.build().rollout("a", 5)


def test_the_device_choice_picks_where_the_training_runs(corpus, monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    mlp_captured, gpt_captured = {}, {}
    _stub_trainer(monkeypatch, "train_mlp_lm", mlp_captured)
    _stub_trainer(monkeypatch, "train_transformer_lm", gpt_captured)
    mlp = _command(corpus, "windowed mlp", seed=0, mlp_steps=20)
    gpt = _command(corpus, "gpt", seed=0, gpt_steps=20)

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
