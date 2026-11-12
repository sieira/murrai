from typing import ClassVar

import pytest

from murrai.adapters.checkpoint import InMemoryCheckpointStore
from murrai.composition import chat as chat_module
from murrai.composition.chat import chat_commands, chat_stages


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    return tmp_path


def test_chat_stages_list_all_four_models(corpus):
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, gpt_steps=20)
    assert [stage.name for stage in stages] == [
        "counted bigram",
        "neural bigram",
        "windowed mlp",
        "char gpt",
    ]
    counted, neural, mlp, gpt = stages
    assert not counted.trains
    assert neural.trains
    assert mlp.trains
    assert gpt.trains


def _stage(corpus, name, **kwargs):
    return next(s for s in chat_stages(corpus, **kwargs) if s.name == name)


def test_counted_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "counted bigram", seed=0).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_neural_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "neural bigram", seed=0, neural_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_gpt_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "char gpt", seed=0, gpt_steps=20).build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_mlp_session_story_carries_generated_text(corpus):
    session = _stage(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
    session.extend_user("a")
    session.extend_generated("b")
    session.extend_user("a")
    assert session.text == "aba"
    assert session.segments() == [
        ("a", "entered"),
        ("b", "generated"),
        ("a", "entered"),
    ]


def test_no_store_means_no_load_stage(corpus):
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert "load from checkpoint" not in [stage.name for stage in stages]


def test_training_stages_save_a_checkpoint(corpus):
    store = InMemoryCheckpointStore()
    stages = chat_stages(
        corpus,
        seed=0,
        neural_steps=20,
        mlp_steps=20,
        gpt_steps=20,
        store=store,
    )
    next(s for s in stages if s.name == "neural bigram").build()
    assert store.names() == ["neural-bigram"]
    next(s for s in stages if s.name == "windowed mlp").build()
    assert store.names() == ["neural-bigram", "windowed-mlp"]
    next(s for s in stages if s.name == "char gpt").build()
    assert store.names() == ["char-gpt", "neural-bigram", "windowed-mlp"]


def test_the_load_stage_restores_the_trained_model(corpus):
    store = InMemoryCheckpointStore()
    first = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    next(s for s in first if s.name == "neural bigram").build()
    second = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    loader = next(s for s in second if s.name == "load from checkpoint")
    assert not loader.trains
    session = loader.build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


class RecordingLine:
    built: ClassVar[list] = []

    def __init__(self):
        self.events = 0
        self.started = False
        self.stopped = False
        RecordingLine.built.append(self)

    def __call__(self, event):
        self.events += 1

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_training_stages_stream_progress_into_the_eta_line(monkeypatch, corpus):
    RecordingLine.built = []
    monkeypatch.setattr(chat_module, "EtaLine", RecordingLine)

    stage = _stage(corpus, "neural bigram", seed=0, neural_steps=20)
    stage.build()

    assert len(RecordingLine.built) == 1
    line = RecordingLine.built[0]
    assert line.events == 20
    assert line.started
    assert line.stopped


def test_the_counted_stage_stays_quiet(monkeypatch, corpus):
    RecordingLine.built = []
    monkeypatch.setattr(chat_module, "EtaLine", RecordingLine)

    _stage(corpus, "counted bigram", seed=0).build()

    assert RecordingLine.built == []


def test_commands_offer_loading_and_retraining(corpus):
    store = InMemoryCheckpointStore()
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    next(s for s in stages if s.name == "neural bigram").build()

    commands = chat_commands(
        corpus, seed=0, neural_steps=20, mlp_steps=20, gpt_steps=20, store=store
    )

    names = [command.name for command in commands]
    assert "load neural-bigram" in names
    assert "retrain neural bigram" in names
    assert "retrain windowed mlp" in names
    assert "retrain char gpt" in names


def test_the_load_command_builds_a_working_session(corpus):
    store = InMemoryCheckpointStore()
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    next(s for s in stages if s.name == "neural bigram").build()

    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    loader = next(c for c in commands if c.name == "load neural-bigram")
    session = loader.build()

    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)


def test_the_retrain_command_trains_and_saves(corpus):
    store = InMemoryCheckpointStore()

    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    retrainer = next(c for c in commands if c.name == "retrain neural bigram")
    session = retrainer.build()

    assert store.names() == ["neural-bigram"]
    session.extend_user("a")
    assert session.rollout("a", 5)


def test_no_store_means_only_retraining_commands(corpus):
    commands = chat_commands(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert [c.name for c in commands] == [
        "retrain neural bigram",
        "retrain windowed mlp",
        "retrain char gpt",
    ]
