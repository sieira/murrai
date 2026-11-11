import math

import pytest
import torch

from murrai.application.runs import train_transformer_lm
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


def test_training_lowers_loss_on_a_toy_corpus():
    result = train_transformer_lm(
        "ab" * 200, steps=40, batch=8, seed=0, depth=1, dimension=8, max_time=8
    )
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_training_reports_held_out_perplexity():
    result = train_transformer_lm(
        "ab" * 400,
        steps=5,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        holdout=0.2,
    )
    assert math.isfinite(result.held_out_perplexity)
    assert result.held_out_perplexity >= 1.0


def test_the_plateau_schedule_shrinks_the_rate_during_a_stalled_run():
    result = train_transformer_lm(
        "abab" * 50,
        steps=60,
        batch=8,
        seed=0,
        depth=1,
        dimension=8,
        max_time=8,
        learning_rate=0.1,
        patience=3,
    )
    assert result.rate_curve[-1] < result.rate_curve[0]
