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
    torch.testing.assert_close(b.probs([0]), torch.tensor([0.5, 0.5]))


def test_rows_sum_to_one(v):
    b = Bigram.fit("aab", v)
    for i in range(v.size):
        torch.testing.assert_close(b.probs([i]).sum(), torch.tensor(1.0))
    torch.testing.assert_close(b.probs([]).sum(), torch.tensor(1.0))


def test_start_row_counts_first_char(v):
    b = Bigram.fit("aab", v)
    torch.testing.assert_close(b.probs([]), torch.tensor([2 / 3, 1 / 3]))


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


def test_fit_reports_progress_chunk_by_chunk():
    text = "aabbaabb" * 50
    marks = []
    Bigram.fit(
        text,
        Vocabulary.from_text("ab"),
        on_progress=lambda counted, total: marks.append((counted, total)),
    )
    assert marks[-1] == (len(text), len(text))
    counted = [counted for counted, _ in marks]
    assert counted == sorted(counted)
    assert all(total == len(text) for _, total in marks)
