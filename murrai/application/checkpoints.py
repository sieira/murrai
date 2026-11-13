from dataclasses import dataclass, field

from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM


@dataclass
class Snapshot:
    kind: str
    merges: tuple
    state: dict
    info: dict = field(default_factory=dict)


def snapshot_of(model, info=None):
    return Snapshot(
        kind=model.kind,
        merges=model.tokenizer.merges,
        state=model.state(),
        info=info or {},
    )


def restore(snapshot: Snapshot):
    for cls in (Bigram, NeuralBigram, MlpLM, TransformerLM):
        if cls.kind == snapshot.kind:
            return cls.from_state(
                TokenizerEncoderDecoder(snapshot.merges),
                snapshot.state,
            )
    raise ValueError(f"unknown checkpoint kind {snapshot.kind!r}")
