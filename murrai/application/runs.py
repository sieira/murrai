from dataclasses import dataclass

from murrai.application.ports import DataSource
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary


@dataclass
class TrainingResult:
    model: Bigram
    train_chars: int
    held_out_chars: int
    held_out_perplexity: float


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
