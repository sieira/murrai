import math

import pytest
import torch

from murrai.domain.mlp import MlpLM
from murrai.domain.vocabulary import Vocabulary


@pytest.fixture
def vocabulary():
    return Vocabulary.from_text("ab")


@pytest.fixture
def model(vocabulary):
    torch.manual_seed(0)
    return MlpLM(vocabulary, window=2, dimension=8, hidden=16)


def test_fresh_model_answers_the_port(model, vocabulary):
    row = model.probs([0])
    assert row.shape == (vocabulary.size,)
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_context_enters_the_model(model):
    assert not torch.allclose(model.probs([0]), model.probs([1, 0]))


def test_window_trims_to_the_last_k(model):
    assert torch.allclose(model.probs([1, 1, 0]), model.probs([1, 0]))


def test_empty_history_uses_the_start(model):
    row = model.probs([])
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)


def test_generation_is_deterministic_and_in_vocabulary(model):
    first = model.generate(12, seed=3)
    assert first == model.generate(12, seed=3)
    assert set(first) <= set(model.vocabulary.chars)


def test_held_out_perplexity_is_finite_and_above_one(model):
    value = model.perplexity("abba")
    assert math.isfinite(value)
    assert value >= 1.0
