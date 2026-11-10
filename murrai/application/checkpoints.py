from dataclasses import dataclass

from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.transformer import TransformerLM
from murrai.domain.vocabulary import Vocabulary


@dataclass
class Snapshot:
    kind: str
    chars: tuple
    state: dict


def snapshot_of(model):
    return Snapshot(kind=model.kind, chars=model.vocabulary.chars, state=model.state())


def restore(snapshot: Snapshot):
    for cls in (Bigram, NeuralBigram, MlpLM, TransformerLM):
        if cls.kind == snapshot.kind:
            return cls.from_state(Vocabulary(snapshot.chars), snapshot.state)
    raise ValueError(f"unknown checkpoint kind {snapshot.kind!r}")
