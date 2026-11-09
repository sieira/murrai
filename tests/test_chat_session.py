import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession


@pytest.fixture
def model():
    return Bigram.fit("aab", Vocabulary.from_text("ab"))


def test_turn_returns_dice_and_rolled(model):
    session = ChatSession(model, seed=0)
    turn = session.turn("a")
    probabilities = [p for _, p in turn.dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0
    assert turn.rolled in ("a", "b")
    assert turn.history == "a"


def test_turns_accumulate_into_history(model):
    session = ChatSession(model, seed=0)
    session.turn("a")
    turn = session.turn("b")
    assert turn.history == "ab"


def test_turn_unknown_char_raises(model):
    session = ChatSession(model, seed=0)
    with pytest.raises(ValueError):
        session.turn("z")


class Recording:
    vocabulary = Vocabulary.from_text("ab")

    def __init__(self):
        self.contexts = []

    def probs(self, context_ids):
        self.contexts.append(list(context_ids))
        return [0.5, 0.5]


def test_session_feeds_the_whole_history_to_the_port():
    recorder = Recording()
    session = ChatSession(recorder, seed=0)
    session.turn("a")
    session.turn("b")
    assert recorder.contexts == [[0], [0, 1]]


@pytest.fixture
def neural_model():
    return train_neural_bigram("abab", steps=3, batch=4, seed=0).model


def test_session_serves_the_neural_bigram_the_same_way(neural_model):
    session = ChatSession(neural_model, seed=0)
    turn = session.turn("a")
    probabilities = [p for _, p in turn.dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0


def test_same_seed_same_rolls(model, neural_model):
    turns = [ChatSession(m, seed=7) for m in (model, neural_model)]
    first = [session.turn("a").rolled for session in turns]
    second = [ChatSession(m, seed=7).turn("a").rolled for m in (model, neural_model)]
    assert first == second
