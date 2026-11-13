import math
from collections import Counter
from pathlib import Path

import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder

A = ord("a")
B = ord("b")


def test_neural_bigram_fresh_predicts_uniform_over_next_tokens():
    model = NeuralBigram(TokenizerEncoderDecoder())
    assert model.probs([0]) == pytest.approx([1 / 256] * 256)


def test_neural_bigram_batch_loss_is_the_nll_of_real_next_tokens():
    model = NeuralBigram(TokenizerEncoderDecoder())
    loss = model.batch_loss([A], [B])
    assert loss.data == pytest.approx(-math.log(1 / 256))


def _aesop():
    text = Path("data/corpus/aesop.txt").read_text()
    cut = int(len(text) * 0.8)
    return text[:cut], text[cut:]


def test_training_neural_bigram_lowers_loss_below_the_counted_benchmark():
    train, _ = _aesop()
    result = train_neural_bigram(train, steps=20, batch=32, seed=0)
    counted_fresh_loss = math.log(result.model.tokenizer.vocab_size)
    assert result.loss_curve[0] == pytest.approx(counted_fresh_loss, abs=0.15)
    assert result.loss_curve[-1] < result.loss_curve[0]


def _row_distances(model, counted, tokenizer, frequent):
    distances = []
    for prev in frequent:
        counted_probs = counted.probs([prev]).tolist()
        learned = model.probs([prev])
        distances.append(
            0.5 * sum(abs(a - b) for a, b in zip(learned, counted_probs, strict=True)),
        )
    return distances


def test_neural_bigram_matches_the_counted_table():
    train, _ = _aesop()
    train = train[:20_000]
    tokenizer = TokenizerEncoderDecoder().fit(train, 300)
    counted = Bigram.fit(train, tokenizer)
    ids = tokenizer.encode(train)
    pairs = zip([tokenizer.vocab_size, *ids][:-1], ids, strict=True)
    prev_counts = Counter(p for p, _ in pairs)
    frequent = [
        prev for prev in range(tokenizer.vocab_size) if prev_counts.get(prev, 0) >= 50
    ]

    result = train_neural_bigram(train, steps=1000, batch=64, seed=0, learning_rate=3.0)
    assert result.loss_curve[-1] < math.log(tokenizer.vocab_size)

    trained = _row_distances(result.model, counted, tokenizer, frequent)
    uniform = _row_distances(NeuralBigram(tokenizer), counted, tokenizer, frequent)
    assert max(trained) < max(uniform) / 2
    assert sum(trained) / len(trained) < sum(uniform) / len(uniform) / 2


def test_neural_bigram_generation_is_deterministic_and_decodes():
    model = NeuralBigram(TokenizerEncoderDecoder())
    first = model.generate(8, seed=3)
    assert first == model.generate(8, seed=3)
    assert len(first) >= 8
