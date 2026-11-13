import pickle
from types import SimpleNamespace

import pytest
import torch

import murrai.composition.chat as murrai_composition_chat
from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from murrai.application.checkpoints import Snapshot
from murrai.application.ports import TrainingEvent
from murrai.application.runs import training_device
from murrai.composition.chat import chat_commands
from murrai.domain.transformer import TransformerLM


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    return tmp_path


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
        "retrain char gpt",
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
    session = _command(corpus, "retrain char gpt", seed=0, gpt_steps=20).build()
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
    assert store.names() == ["counted-bigram"]
    _command(corpus, "retrain neural bigram", store=store, neural_steps=20).build()
    assert store.names() == ["counted-bigram", "neural-bigram"]
    _command(corpus, "retrain windowed mlp", store=store, mlp_steps=20).build()
    assert store.names() == ["counted-bigram", "neural-bigram", "windowed-mlp"]
    _command(corpus, "retrain char gpt", store=store, gpt_steps=20).build()
    assert store.names() == [
        "char-gpt",
        "counted-bigram",
        "neural-bigram",
        "windowed-mlp",
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
    ).build()

    info = store.load("neural-bigram").info
    assert set(info) == {
        "saved_at",
        "corpus",
        "seed",
        "steps",
        "batch",
        "learning_rate",
        "final_loss",
    }
    assert info["saved_at"] == "2027-01-15T08:00:00+00:00"
    assert info["corpus"] == str(corpus)
    assert info["seed"] == 0
    assert info["steps"] == 20
    assert info["batch"] == 64
    assert info["learning_rate"] == 3.0
    assert isinstance(info["final_loss"], float)


def test_the_counted_checkpoint_records_no_steps(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "counted bigram",
        seed=0,
        store=store,
        clock=_clock_at_2027,
    ).build()

    info = store.load("counted-bigram").info
    assert set(info) == {"saved_at", "corpus", "seed", "held_out_perplexity"}
    assert info["saved_at"] == "2027-01-15T08:00:00+00:00"
    assert isinstance(info["held_out_perplexity"], float)


def test_the_mlp_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain windowed mlp",
        seed=0,
        mlp_steps=20,
        store=store,
        clock=_clock_at_2027,
    ).build()

    info = store.load("windowed-mlp").info
    assert set(info) == {
        "saved_at",
        "corpus",
        "seed",
        "steps",
        "batch",
        "learning_rate",
        "final_loss",
        "held_out_perplexity",
    }
    assert info["learning_rate"] == 0.5
    assert info["steps"] == 20
    assert info["batch"] == 256


def test_the_gpt_checkpoint_records_its_info(corpus):
    store = InMemoryCheckpointStore()
    _command(
        corpus,
        "retrain char gpt",
        seed=0,
        gpt_steps=20,
        store=store,
        clock=_clock_at_2027,
    ).build()

    info = store.load("char-gpt").info
    assert set(info) == {
        "saved_at",
        "corpus",
        "seed",
        "steps",
        "batch",
        "learning_rate",
        "final_loss",
        "held_out_perplexity",
    }
    assert info["learning_rate"] == 0.001


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

    info = store.load("neural-bigram").info
    assert info["final_loss"] == captured["result"].loss_curve[-1]


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
    info = store.load("neural-bigram").info

    loader = _command(
        corpus,
        "load neural-bigram",
        seed=0,
        store=store,
        clock=_clock_at_2027,
    )
    assert loader.description == (
        f"saved 2027-01-15 08:00 · 20 steps · loss {info['final_loss']:.2f}"
    )


def test_a_checkpoint_without_info_keeps_the_old_load_line(corpus):
    store = InMemoryCheckpointStore()
    store.save(
        "neural-bigram",
        Snapshot(kind="neural-bigram", chars=("a", "b"), state={}),
    )
    loader = _command(corpus, "load neural-bigram", seed=0, store=store)
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

    loader = _command(corpus, "load neural-bigram", seed=0, store=store)
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

    assert len(events) == 20
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
    assert "load neural-bigram" in names
    assert "counted bigram" in names
    assert "retrain neural bigram" in names
    assert "retrain windowed mlp" in names
    assert "retrain char gpt" in names


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

    loader = _command(corpus, "load neural-bigram", seed=0, store=store)
    session = loader.build()

    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_training_saves_the_model_before_the_metrics_run(tmp_path, monkeypatch):
    def broken_perplexity(self, text):
        raise RuntimeError("metrics blew up")

    monkeypatch.setattr(TransformerLM, "perplexity", broken_perplexity)
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    store = FileCheckpointStore(tmp_path / "checkpoints")
    command = _command(tmp_path, "retrain char gpt", seed=0, gpt_steps=5, store=store)
    with pytest.raises(RuntimeError):
        command.build()
    assert store.names() == ["char-gpt"]


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

    assert store.names() == ["neural-bigram"]
    session.extend_user("a")
    assert session.rollout("a", 5)


def test_no_store_means_counted_and_retraining_commands(corpus):
    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert [command.name for command in commands] == [
        "counted bigram",
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain char gpt",
    ]


def test_training_commands_survive_a_trip_to_another_process(corpus):
    import pickle

    commands = chat_commands(corpus, seed=0, neural_steps=20)
    trains = [c for c in commands if c.trains]
    assert [c.name for c in trains] == [
        "counted bigram",
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain char gpt",
    ]

    for command in trains:
        assert pickle.loads(pickle.dumps(command)).name == command.name

    revived = pickle.loads(pickle.dumps(trains[0]))
    assert revived.build().rollout("a", 5)


def test_the_gpt_command_trains_on_the_policy_device(corpus, monkeypatch):
    real = murrai_composition_chat.train_transformer_lm
    captured = {}

    def recorder(text, **kwargs):
        captured.update(kwargs)
        return real(text, **kwargs)

    monkeypatch.setattr(murrai_composition_chat, "train_transformer_lm", recorder)
    _command(corpus, "retrain char gpt", seed=0, gpt_steps=20).build()
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
    ]
    assert counted.fields[-1].check == "from zero below one"


def test_the_neural_command_offers_its_knobs(corpus):
    fields = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20).fields

    assert [(f.name, f.label, f.kind, f.default) for f in fields] == [
        ("text", "text", "choice", corpus.name),
        ("size", "size", "choice", "all"),
        ("steps", "steps", "int", 20),
        ("batch", "batch", "int", 64),
        ("learning_rate", "learning rate", "float", 3.0),
        ("final_learning_rate", "final learning rate", "float", 0.1),
        ("seed", "seed", "int", 0),
        ("start_from", "start from", "choice", "fresh"),
    ]
    assert fields[-1].choices == ("fresh",)


def test_the_start_from_choice_lists_the_saved_models(corpus):
    store = InMemoryCheckpointStore()
    store.save("char-gpt", Snapshot(kind="char-gpt", chars=("a", "b"), state={}))
    store.save(
        "neural-bigram",
        Snapshot(kind="neural-bigram", chars=("a", "b"), state={}),
    )

    field = _field(
        corpus,
        "retrain neural bigram",
        "start_from",
        seed=0,
        neural_steps=20,
        store=store,
    )

    assert field.choices == ("fresh", "char-gpt", "neural-bigram")


def test_the_mlp_command_offers_its_knobs(corpus):
    fields = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
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
        ("start_from", "start from", "fresh"),
    ]
    assert [f.name for f in fields if f.shape] == ["window", "dimension", "hidden"]
    assert fields[-2].choices == ("auto", "prefer gpu", "cpu")


def test_the_gpt_command_offers_its_knobs(corpus):
    fields = _command(corpus, "retrain char gpt", seed=0, gpt_steps=20).fields

    assert [(f.name, f.label, f.default) for f in fields] == [
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
        ("start_from", "start from", "fresh"),
    ]
    assert [f.name for f in fields if f.shape] == [
        "depth",
        "dimension",
        "heads",
        "max_time",
    ]
    assert fields[-2].choices == ("auto", "prefer gpu", "cpu")


def test_the_baked_command_carries_the_values(corpus):
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)
    values = {**_defaults(command), "steps": 5}

    baked = command.bake(values)

    assert baked.given == values
    assert baked.name == command.name
    assert baked.trains


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

    baked = trainer.bake({**_defaults(trainer), "start_from": "neural-bigram"})
    assert baked.given["start_from"] == "neural-bigram"
    assert "start" not in baked.given

    baked.build()
    assert captured["kwargs"]["start"].kind == "neural-bigram"


def test_a_fresh_start_hands_the_trainer_no_snapshot(corpus, monkeypatch):
    captured = _capturing(monkeypatch, "train_neural_bigram")
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)

    command.bake(_defaults(command)).build()

    assert captured["kwargs"]["start"] is None


def test_a_saved_start_leaves_the_shape_to_the_savegame(corpus, monkeypatch):
    gpt_captured = _capturing(monkeypatch, "train_transformer_lm")
    mlp_captured = _capturing(monkeypatch, "train_mlp_lm")
    store = InMemoryCheckpointStore()
    gpt = _command(corpus, "retrain char gpt", seed=0, gpt_steps=20, store=store)
    mlp = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20, store=store)
    gpt.build()
    mlp.build()

    gpt.bake(
        {**_defaults(gpt), "start_from": "char-gpt", "depth": 9, "max_time": 128},
    ).build()
    mlp.bake(
        {**_defaults(mlp), "start_from": "windowed-mlp", "window": 3, "hidden": 9},
    ).build()

    for captured in (gpt_captured, mlp_captured):
        assert captured["kwargs"]["start"].kind in ("char-gpt", "windowed-mlp")
        for shape in ("depth", "dimension", "heads", "max_time", "window", "hidden"):
            assert shape not in captured["kwargs"]


def test_a_baked_command_survives_the_trip_to_another_process(corpus):
    command = _command(corpus, "retrain neural bigram", seed=0, neural_steps=20)
    baked = command.bake({**_defaults(command), "steps": 5})

    revived = pickle.loads(pickle.dumps(baked))

    assert revived.given == baked.given
    assert revived.build().rollout("a", 5)


def test_the_device_choice_picks_where_the_training_runs(corpus, monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    mlp_captured, gpt_captured = {}, {}
    _stub_trainer(monkeypatch, "train_mlp_lm", mlp_captured)
    _stub_trainer(monkeypatch, "train_transformer_lm", gpt_captured)
    mlp = _command(corpus, "retrain windowed mlp", seed=0, mlp_steps=20)
    gpt = _command(corpus, "retrain char gpt", seed=0, gpt_steps=20)

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
