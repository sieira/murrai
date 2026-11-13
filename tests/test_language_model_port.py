import pytest
import torch

from murrai.application.ports import LanguageModel
from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.transformer import TransformerLM
from murrai.domain.tokenizer import TokenizerEncoderDecoder


@pytest.fixture
def counted():
    return Bigram.fit("abab", TokenizerEncoderDecoder())


@pytest.fixture
def neural():
    return train_neural_bigram("abab", steps=5, batch=8, seed=0).model


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MlpLM(TokenizerEncoderDecoder(), window=2, dimension=8, hidden=16)


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return TransformerLM(
        TokenizerEncoderDecoder(),
        depth=2,
        dimension=8,
        max_time=4,
    )


def test_all_models_satisfy_the_language_model_port(counted, neural, mlp, transformer):
    for model in (counted, neural, mlp, transformer):
        assert isinstance(model, LanguageModel)


def test_a_one_char_history_conditions_the_answer(counted):
    expected = torch.full((256,), 1 / 258)
    expected[ord("b")] = 3 / 258
    torch.testing.assert_close(counted.probs([ord("a")]), expected)


def test_an_empty_history_means_the_start(counted):
    expected = torch.full((256,), 1 / 257)
    expected[ord("a")] = 2 / 257
    torch.testing.assert_close(counted.probs([]), expected)


def test_the_counted_bigram_uses_the_last_char_only(counted):
    torch.testing.assert_close(counted.probs([ord("b"), ord("a")]), counted.probs([ord("a")]))
    expected = torch.full((256,), 1 / 257)
    expected[ord("a")] = 2 / 257
    torch.testing.assert_close(counted.probs([ord("b"), ord("a"), ord("b")]), expected)


def test_the_neural_bigram_answers_the_port(neural):
    assert neural.probs([1, 0]) == pytest.approx(neural.probs([0]))
    assert sum(neural.probs([1, 0])) == pytest.approx(1.0)
    assert sum(neural.probs([])) == pytest.approx(1.0)
