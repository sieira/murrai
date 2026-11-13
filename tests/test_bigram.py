import math

import pytest
import torch

from murrai.domain.bigram import Bigram
from murrai.domain.ports import Progress
from murrai.domain.tokenizer import TokenizerEncoderDecoder

A = ord("a")
B = ord("b")


@pytest.fixture
def tokenizer():
    return TokenizerEncoderDecoder()


def test_fit_uses_smoothing(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    expected = torch.full((tokenizer.vocab_size,), 1 / 258)
    expected[A] = 2 / 258
    expected[B] = 2 / 258
    torch.testing.assert_close(b.probs([A]), expected)


def test_rows_sum_to_one(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    for i in range(tokenizer.vocab_size):
        torch.testing.assert_close(b.probs([i]).sum(), torch.tensor(1.0))
    torch.testing.assert_close(b.probs([]).sum(), torch.tensor(1.0))


def test_start_row_counts_first_char(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    expected = torch.full((tokenizer.vocab_size,), 1 / 257)
    expected[A] = 2 / 257
    torch.testing.assert_close(b.probs([]), expected)


def test_nll_and_perplexity_hand_computed(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    expected = -(math.log(2 / 257) + math.log(2 / 258)) / 2
    assert b.nll("ab") == pytest.approx(expected)
    assert b.perplexity("ab") == pytest.approx(math.exp(expected))


def test_same_seed_same_generation(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    assert b.generate(20, seed=0) == b.generate(20, seed=0)


def test_different_seed_differs(tokenizer):
    b = Bigram.fit("aab", tokenizer)
    assert b.generate(20, seed=0) != b.generate(20, seed=1)


def test_fit_reports_progress_chunk_by_chunk():
    text = "aabbaabb" * 50
    bus = FakeBus()
    Bigram.fit(text, TokenizerEncoderDecoder(), sink=bus)
    marks = [(event.step, event.total_steps) for event in bus.events]
    assert marks[-1] == (len(text), len(text))
    counted = [counted for counted, _ in marks]
    assert counted == sorted(counted)
    assert all(total == len(text) for _, total in marks)
    assert all(isinstance(event, Progress) for event in bus.events)


class FakeBus:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)
