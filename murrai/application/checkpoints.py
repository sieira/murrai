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
class RunSettings:
    saved_at: str
    corpus: str
    seed: int
    corpus_chars: int | None = None


@dataclass(frozen=True)
class GradientRun:
    steps: int
    batch: int
    learning_rate: float
    final_loss: float | None = None


@dataclass(frozen=True)
class CheckpointRecord:
    settings: RunSettings | None = None
    gradient: GradientRun | None = None
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
