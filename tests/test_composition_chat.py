import pytest

import murrai.composition.chat as murrai_composition_chat
from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
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
    _command(corpus, "retrain gpt", store=store, gpt_steps=20).build()
    assert store.names() == [
        "counted-bigram",
        "gpt",
        "neural-bigram",
        "windowed-mlp",
    ]


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
