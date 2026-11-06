import random
from dataclasses import dataclass

from murrai.application.ports import DataSource
from murrai.domain.bigram import Bigram
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.vocabulary import Vocabulary


@dataclass
class TrainingResult:
    model: Bigram
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float


@dataclass
class NeuralTrainingResult:
    model: NeuralBigram
    loss_curve: list[float]


def train_bigram(source: DataSource, holdout: float) -> TrainingResult:
    text = source.text()
    cut = int(len(text) * (1 - holdout))
    train, held_out = text[:cut], text[cut:]

    vocabulary = Vocabulary.from_text(text)
    model = Bigram.fit(train, vocabulary)
    return TrainingResult(
        model=model,
        train_chars=len(train),
        held_out_chars=len(held_out),
        held_out_perplexity=model.perplexity(held_out),
    )


def train_neural_bigram(
    train, steps, batch, seed, learning_rate=0.5, final_learning_rate=0.1
):
    vocabulary = Vocabulary.from_text(train)
    model = NeuralBigram(vocabulary)
    ids = [vocabulary.size, *vocabulary.encode(train)]
    pairs = [(prev, nxt) for prev, nxt in zip(ids, ids[1:])]

    rng = random.Random(seed)
    loss_curve = []
    for step in range(steps):
        rate = learning_rate + (final_learning_rate - learning_rate) * step / (
            steps - 1
        )
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(batch)]
        prevs = [prev for prev, _ in sample]
        nexts = [nxt for _, nxt in sample]
        loss = model.batch_loss(prevs, nexts)
        loss.backward()
        model.update(rate)
        loss_curve.append(loss.data)
    return NeuralTrainingResult(model=model, loss_curve=loss_curve)
