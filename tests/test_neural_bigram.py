import math
from collections import Counter
from pathlib import Path

import pytest

from murrai.application.checkpoints import snapshot_of
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


def _row_distances(model, counted, vocabulary, frequent):
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
    vocabulary = Vocabulary.from_text(train)
    counted = Bigram.fit(train, vocabulary)
    ids = vocabulary.encode(train)
    pairs = zip([vocabulary.size, *ids][:-1], ids, strict=True)
    prev_counts = Counter(p for p, _ in pairs)
    frequent = [
        prev for prev in range(vocabulary.size) if prev_counts.get(prev, 0) >= 50
    ]

    result = train_neural_bigram(train, steps=1000, batch=64, seed=0, learning_rate=3.0)
    assert result.loss_curve[-1] < math.log(vocabulary.size)

    trained = _row_distances(result.model, counted, vocabulary, frequent)
    uniform = _row_distances(NeuralBigram(vocabulary), counted, vocabulary, frequent)
    assert max(trained) < max(uniform) / 2
    assert sum(trained) / len(trained) < sum(uniform) / len(uniform) / 2


def test_neural_bigram_generation_is_deterministic_and_in_vocabulary():
    vocabulary = Vocabulary.from_text("abab")
    model = NeuralBigram(vocabulary)
    first = model.generate(8, seed=3)
    assert first == model.generate(8, seed=3)
    assert set(first) <= set(vocabulary.chars)


def test_growth_fits_the_new_vocabulary():
    model = NeuralBigram(Vocabulary.from_text("ab"))
    grown = model.grown_to(Vocabulary.from_text("abc"))
    assert grown.vocabulary.chars == ("a", "b", "c")
    assert grown.start == 3
    assert len(grown.weights) == 4
    assert all(len(row) == 3 for row in grown.weights)


def test_growth_keeps_old_rows_and_moves_the_start_row():
    model = NeuralBigram(Vocabulary.from_text("ab"))
    model.weights[0][1].data = 2.0
    model.weights[1][0].data = -0.5
    model.weights[2][0].data = -1.0
    grown = model.grown_to(Vocabulary.from_text("abc"))
    assert [v.data for v in grown.logits(0)] == [0.0, 2.0, 0.0]
    assert [v.data for v in grown.logits(1)] == [-0.5, 0.0, 0.0]
    assert [v.data for v in grown.logits(3)] == [-1.0, 0.0, 0.0]


def test_growth_gives_new_characters_untrained_zero_rows():
    model = NeuralBigram(Vocabulary.from_text("ab"))
    grown = model.grown_to(Vocabulary.from_text("abc"))
    assert [v.data for v in grown.logits(2)] == [0.0, 0.0, 0.0]


def test_training_continues_from_a_snapshot():
    first = train_neural_bigram("ab" * 400, steps=60, batch=64, seed=0)
    second = train_neural_bigram(
        "ab" * 400,
        steps=60,
        batch=64,
        seed=0,
        start=snapshot_of(first.model),
    )
    assert second.loss_curve[0] < first.loss_curve[0]
    assert second.loss_curve[0] == pytest.approx(first.loss_curve[-1], abs=0.01)
    assert second.loss_curve[-1] < second.loss_curve[0]


def test_training_continuation_grows_the_vocabulary():
    first = train_neural_bigram("ab" * 400, steps=60, batch=64, seed=0)
    second = train_neural_bigram(
        "ab" * 400 + "c" * 200,
        steps=60,
        batch=64,
        seed=0,
        start=snapshot_of(first.model),
    )
    assert second.model.vocabulary.chars == ("a", "b", "c")
    assert second.loss_curve[0] < math.log(3)


def test_continuing_from_an_untrained_snapshot_repeats_the_fresh_run():
    fresh = train_neural_bigram("ab" * 400, steps=20, batch=64, seed=0)
    repeated = train_neural_bigram(
        "ab" * 400,
        steps=20,
        batch=64,
        seed=0,
        start=snapshot_of(NeuralBigram(Vocabulary.from_text("ab" * 400))),
    )
    assert repeated.loss_curve == fresh.loss_curve
