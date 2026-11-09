import pytest

from murrai.adapters.checkpoint import InMemoryCheckpointStore
from murrai.composition.chat import chat_stages


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb")
    return tmp_path


def test_chat_stages_list_both_models(corpus):
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20)
    assert [stage.name for stage in stages] == [
        "counted bigram",
        "neural bigram",
        "windowed mlp",
    ]
    counted, neural, mlp = stages
    assert not counted.trains
    assert neural.trains
    assert mlp.trains


def _stage(corpus, name, **kwargs):
    return [s for s in chat_stages(corpus, **kwargs) if s.name == name][0]


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
    stages = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    [s for s in stages if s.name == "neural bigram"][0].build()
    assert store.names() == ["neural-bigram"]
    [s for s in stages if s.name == "windowed mlp"][0].build()
    assert store.names() == ["neural-bigram", "windowed-mlp"]


def test_the_load_stage_restores_the_trained_model(corpus):
    store = InMemoryCheckpointStore()
    first = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    [s for s in first if s.name == "neural bigram"][0].build()
    second = chat_stages(corpus, seed=0, neural_steps=20, mlp_steps=20, store=store)
    loader = [s for s in second if s.name == "load from checkpoint"][0]
    assert not loader.trains
    session = loader.build()
    session.extend_user("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.rollout("a", 5)
