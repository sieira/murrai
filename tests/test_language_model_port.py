import pytest
import torch

from murrai.application.ports import LanguageModel
from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary


@pytest.fixture
def counted():
    return Bigram.fit("abab", Vocabulary.from_text("ab"))


@pytest.fixture
def neural():
    return train_neural_bigram("abab", steps=5, batch=8, seed=0).model


def test_both_models_satisfy_the_language_model_port(counted, neural):
    for model in (counted, neural):
        assert isinstance(model, LanguageModel)


def test_a_one_char_history_conditions_the_answer(counted):
    torch.testing.assert_close(counted.probs([0]), torch.tensor([0.25, 0.75]))


def test_an_empty_history_means_the_start(counted):
    torch.testing.assert_close(counted.probs([]), torch.tensor([2 / 3, 1 / 3]))


def test_the_counted_bigram_uses_the_last_char_only(counted):
    torch.testing.assert_close(counted.probs([1, 0]), counted.probs([0]))
    torch.testing.assert_close(counted.probs([1, 0, 1]), torch.tensor([2 / 3, 1 / 3]))


def test_the_neural_bigram_answers_the_port(neural):
    assert neural.probs([1, 0]) == pytest.approx(neural.probs([0]))
    assert sum(neural.probs([1, 0])) == pytest.approx(1.0)
    assert sum(neural.probs([])) == pytest.approx(1.0)
