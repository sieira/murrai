from dataclasses import dataclass

import torch

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


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value


def snapshot_of(model):
    return Snapshot(
        kind=model.kind,
        merges=model.tokenizer.merges,
        state=_clone(model.state()),
    )


def run_snapshot(model, run_state):
    snapshot = snapshot_of(model)
    return Snapshot(
        kind=snapshot.kind,
        merges=snapshot.merges,
        state={**snapshot.state, "run": _clone(run_state)},
    )


def restore(snapshot: Snapshot):
    for cls in (Bigram, NeuralBigram, MlpLM, TransformerLM):
        if cls.kind == snapshot.kind:
            return cls.from_state(
                TokenizerEncoderDecoder(snapshot.merges),
                snapshot.state,
            )
    raise ValueError(f"unknown checkpoint kind {snapshot.kind!r}")
