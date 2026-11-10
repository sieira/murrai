import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession


@pytest.fixture
def model():
    return Bigram.fit("abab", Vocabulary.from_text("ab"))


def test_the_story_holds_entered_and_generated_text(model):
    session = ChatSession(model, seed=0)
    session.commit("ab")
    rolled = session.send()
    assert session.text == "ab" + rolled
    assert rolled in ("a", "b")


def test_commit_returns_only_the_user_addition(model):
    session = ChatSession(model, seed=0)
    session.commit("a")
    session.send()
    delta = session.commit(session.text + "b")
    assert delta == "b"


def test_commit_validates_against_the_vocabulary(model):
    session = ChatSession(model, seed=0)
    with pytest.raises(ValueError):
        session.commit("az")


def test_the_die_is_sorted_and_sums_to_one(model):
    session = ChatSession(model, seed=0)
    session.commit("a")
    dice = session.dice()
    probabilities = [p for _, p in dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0


def test_segments_mark_entered_and_generated_runs(model):
    session = ChatSession(model, seed=0)
    session.commit("ab")
    rolled = session.send()
    assert session.segments() == [("ab", "entered"), (rolled, "generated")]


def test_send_is_deterministic_per_seed(model):
    first = ChatSession(model, seed=7)
    second = ChatSession(model, seed=7)
    first.commit("a")
    second.commit("a")
    assert first.send() == second.send()


class Recording:
    vocabulary = Vocabulary.from_text("ab")

    def __init__(self):
        self.contexts = []

    def probs(self, context_ids):
        self.contexts.append(list(context_ids))
        return [0.5, 0.5]


def test_send_extends_the_context_with_the_generated_char():
    recorder = Recording()
    session = ChatSession(recorder, seed=0)
    session.commit("a")
    rolled = session.send()
    assert recorder.contexts == [[0]]
    session.dice()
    rolled_id = Recording.vocabulary.chars.index(rolled)
    assert recorder.contexts[-1] == [0, rolled_id]


def test_unknown_chars_in_a_later_commit_raise_without_corrupting(model):
    session = ChatSession(model, seed=0)
    session.commit("a")
    with pytest.raises(ValueError):
        session.commit(session.text + "z")
    assert session.text == "a"


@pytest.fixture
def neural_model():
    return train_neural_bigram("abab", steps=3, batch=4, seed=0).model


def test_session_serves_the_neural_bigram_the_same_way(neural_model):
    session = ChatSession(neural_model, seed=0)
    session.commit("ab")
    dice = session.dice()
    probabilities = [p for _, p in dice]
    assert pytest.approx(sum(probabilities)) == 1.0
