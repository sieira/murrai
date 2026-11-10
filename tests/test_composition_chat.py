import pytest

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
    session.commit("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.send() in ("a", "b")


def test_neural_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "neural bigram", seed=0, neural_steps=20).build()
    session.commit("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.send() in ("a", "b")


def test_mlp_stage_builds_a_working_session(corpus):
    session = _stage(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
    session.commit("a")
    probabilities = [p for _, p in session.dice()]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert session.send() in ("a", "b")


def test_mlp_session_story_carries_generated_text(corpus):
    session = _stage(corpus, "windowed mlp", seed=0, mlp_steps=20).build()
    session.commit("a")
    rolled = session.send()
    session.commit(session.text + "a")
    assert session.text == "a" + rolled + "a"
