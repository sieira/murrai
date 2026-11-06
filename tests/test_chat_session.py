import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession


@pytest.fixture
def model():
    return Bigram.fit("aab", Vocabulary.from_text("ab"))


def test_dice_sorted_descending(model):
    session = ChatSession(model, seed=0)
    dice = session.dice("a")
    assert session.dice("a") == dice
    probabilities = [p for _, p in dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0


def test_roll_deterministic_per_seed(model):
    assert ChatSession(model, seed=0).roll("a") == ChatSession(model, seed=0).roll("a")


def test_unknown_char_raises(model):
    session = ChatSession(model, seed=0)
    with pytest.raises(ValueError):
        session.dice("z")


@pytest.fixture
def neural_model():
    return train_neural_bigram("abab", steps=3, batch=4, seed=0).model


def test_session_serves_the_neural_bigram_the_same_way(neural_model):
    session = ChatSession(neural_model, seed=0)
    dice = session.dice("a")
    assert session.dice("a") == dice
    probabilities = [p for _, p in dice]
    assert probabilities == sorted(probabilities, reverse=True)
    assert pytest.approx(sum(probabilities)) == 1.0
    assert ChatSession(neural_model, seed=0).roll("a") == session.roll("a")


def test_neural_session_unknown_char_raises(neural_model):
    session = ChatSession(neural_model, seed=0)
    with pytest.raises(ValueError):
        session.dice("z")
