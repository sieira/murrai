import math

import pytest
import torch

from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary


@pytest.fixture
def v():
    return Vocabulary.from_text("ab")


def test_fit_uses_smoothing(v):
    b = Bigram.fit("aab", v)
    torch.testing.assert_close(b.probs(0), torch.tensor([0.5, 0.5]))


def test_rows_sum_to_one(v):
    b = Bigram.fit("aab", v)
    for i in range(v.size + 1):
        torch.testing.assert_close(b.probs(i).sum(), torch.tensor(1.0))


def test_start_row_counts_first_char(v):
    b = Bigram.fit("aab", v)
    torch.testing.assert_close(b.probs(v.size), torch.tensor([2 / 3, 1 / 3]))


def test_nll_and_perplexity_hand_computed(v):
    b = Bigram.fit("aab", v)
    expected = -(math.log(2 / 3) + math.log(0.5)) / 2
    assert b.nll("ab") == pytest.approx(expected)
    assert b.perplexity("ab") == pytest.approx(math.exp(expected))


def test_same_seed_same_generation(v):
    b = Bigram.fit("aab", v)
    assert b.generate(20, seed=0) == b.generate(20, seed=0)


def test_different_seed_differs(v):
    b = Bigram.fit("aab", v)
    assert b.generate(20, seed=0) != b.generate(20, seed=1)
