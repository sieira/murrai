import pytest

from murrai.application.session import ChatSession
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary


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
