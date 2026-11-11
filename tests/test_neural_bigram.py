import math
from collections import Counter
from pathlib import Path

import pytest

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.vocabulary import Vocabulary


def test_neural_bigram_fresh_predicts_uniform_over_next_chars():
    vocabulary = Vocabulary.from_text("abab")
    model = NeuralBigram(vocabulary)
    assert model.probs([0]) == pytest.approx([0.5, 0.5])


def test_neural_bigram_batch_loss_is_the_nll_of_real_next_chars():
    vocabulary = Vocabulary.from_text("abab")
    model = NeuralBigram(vocabulary)
    loss = model.batch_loss([0], [1])
    assert loss.data == pytest.approx(-math.log(0.5))


def _aesop():
    text = Path("data/corpus/aesop.txt").read_text()
    cut = int(len(text) * 0.8)
    return text[:cut], text[cut:]


def test_training_neural_bigram_lowers_loss_below_the_counted_benchmark():
    train, _ = _aesop()
    result = train_neural_bigram(train, steps=20, batch=32, seed=0)
    counted_fresh_loss = math.log(117)
    assert result.loss_curve[0] == pytest.approx(counted_fresh_loss, abs=0.15)
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_neural_bigram_matches_the_counted_table():
    train, _ = _aesop()
    vocabulary = Vocabulary.from_text(train)
    result = train_neural_bigram(train, steps=2000, batch=64, seed=0, learning_rate=3.0)
    neural = result.model
    counted = Bigram.fit(train, vocabulary)
    ids = vocabulary.encode(train)
    pairs = zip([vocabulary.size, *ids][:-1], ids, strict=False)
    prev_counts = Counter(p for p, _ in pairs)

    assert result.loss_curve[-1] < 2.90

    distances = []
    for prev in range(vocabulary.size):
        if prev_counts.get(prev, 0) < 50:
            continue
        counted_probs = counted.probs([prev]).tolist()
        learned = neural.probs([prev])
        distance = 0.5 * sum(
            abs(a - b) for a, b in zip(learned, counted_probs, strict=False)
        )
        distances.append(distance)

    assert max(distances) < 0.45
    assert sum(distances) / len(distances) < 0.25


def test_neural_bigram_generation_is_deterministic_and_in_vocabulary():
    vocabulary = Vocabulary.from_text("abab")
    model = NeuralBigram(vocabulary)
    first = model.generate(8, seed=3)
    assert first == model.generate(8, seed=3)
    assert set(first) <= set(vocabulary.chars)
