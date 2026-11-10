import math

import pytest
import torch

from murrai.domain.transformer import TransformerLM
from murrai.domain.vocabulary import Vocabulary


@pytest.fixture
def vocabulary():
    return Vocabulary.from_text("ab")


@pytest.fixture
def model(vocabulary):
    torch.manual_seed(0)
    return TransformerLM(vocabulary, depth=2, dimension=16, max_time=8)


def test_logits_answer_the_vocabulary(model):
    logits = model.logits([0, 1])
    assert logits.shape == (model.vocabulary.size,)


def test_different_order_of_the_same_chars_different_logits(model):
    assert not torch.allclose(model.logits([0, 1]), model.logits([1, 0]))


def test_an_empty_history_answers_through_the_start(model):
    row = model.probs([])
    assert row.shape == (model.vocabulary.size,)
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_generation_wanders_past_the_window(model):
    long = model.generate(3 * model.max_time, seed=3)
    assert long == model.generate(3 * model.max_time, seed=3)
    assert set(long) <= set(model.vocabulary.chars)


def test_the_state_round_trips(model):
    vocabulary = Vocabulary(model.vocabulary.chars)
    rebuilt = TransformerLM.from_state(vocabulary, model.state())
    torch.testing.assert_close(rebuilt.logits([0, 1, 0]), model.logits([0, 1, 0]))


def test_generation_is_deterministic_and_in_vocabulary(model):
    first = model.generate(12, seed=3)
    assert first == model.generate(12, seed=3)
    assert set(first) <= set(model.vocabulary.chars)


def test_nll_matches_a_hand_computed_case(model):
    model.head.weight.data.zero_()
    model.head.bias.data.copy_(torch.tensor([0.0, math.log(3.0)]))
    expected = -(math.log(0.25) + math.log(0.75)) / 2
    assert model.nll("ab") == pytest.approx(expected)
