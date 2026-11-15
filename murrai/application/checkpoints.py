from dataclasses import dataclass

from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM


@dataclass(frozen=True)
class Snapshot:
    kind: str
    merges: tuple
    state: dict


@dataclass(frozen=True)
class CheckpointRecord:
    saved_at: str | None = None
    corpus: str | None = None
    seed: int | None = None
    steps: int | None = None
    batch: int | None = None
    learning_rate: float | None = None
    final_loss: float | None = None
    held_out_perplexity: float | None = None


def snapshot_of(model):
    return Snapshot(
        kind=model.kind,
        merges=model.tokenizer.merges,
        state=model.state(),
    )


def restore(snapshot: Snapshot):
    for cls in (Bigram, NeuralBigram, MlpLM, TransformerLM):
        if cls.kind == snapshot.kind:
            return cls.from_state(
                TokenizerEncoderDecoder(snapshot.merges),
                snapshot.state,
            )
    raise ValueError(f"unknown checkpoint kind {snapshot.kind!r}")
