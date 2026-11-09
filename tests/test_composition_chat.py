import pytest

from murrai.composition.chat import chat_stages


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "story.txt").write_text("aabbaabb")
    return tmp_path


def test_chat_stages_list_both_models(corpus):
    stages = chat_stages(corpus, seed=0, neural_steps=20)
    assert [stage.name for stage in stages] == ["counted bigram", "neural bigram"]
    counted, neural = stages
    assert not counted.trains
    assert neural.trains


def test_counted_stage_builds_a_working_session(corpus):
    (counted,) = [s for s in chat_stages(corpus, seed=0) if s.name == "counted bigram"]
    session = counted.build()
    turn = session.turn("a")
    probabilities = [p for _, p in turn.dice]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert turn.rolled in ("a", "b")


def test_neural_stage_builds_a_working_session(corpus):
    (neural,) = [
        s
        for s in chat_stages(corpus, seed=0, neural_steps=20)
        if s.name == "neural bigram"
    ]
    session = neural.build()
    turn = session.turn("a")
    probabilities = [p for _, p in turn.dice]
    assert pytest.approx(sum(probabilities)) == 1.0
    assert turn.rolled in ("a", "b")
