import copy
from dataclasses import dataclass

import torch

from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.shape import Shape
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM


@dataclass(frozen=True)
class Snapshot:
    kind: str
    merges: tuple
    shape: Shape
    weights: object
    run: RunState | None = None


@dataclass(frozen=True)
class RunState:
    step: int
    rate: float
    best: float
    best_step: int
    train_seconds: float
    optimizer: dict | None
    loss_curve: list[float]
    rate_curve: list[float]
    validation_curve: list[tuple[int, float]]

    def __post_init__(self):
        object.__setattr__(self, "loss_curve", list(self.loss_curve))
        object.__setattr__(self, "rate_curve", list(self.rate_curve))
        object.__setattr__(self, "validation_curve", list(self.validation_curve))
        if self.optimizer is not None:
            object.__setattr__(self, "optimizer", copy.deepcopy(self.optimizer))

    @classmethod
    def fresh(cls, learning_rate):
        return cls(
            step=0,
            rate=learning_rate,
            best=float("inf"),
            best_step=0,
            train_seconds=0.0,
            optimizer=None,
            loss_curve=[],
            rate_curve=[],
            validation_curve=[],
        )

    def to_dict(self):
        return {
            "step": self.step,
            "rate": self.rate,
            "best": self.best,
            "best_step": self.best_step,
            "train_seconds": self.train_seconds,
            "optimizer": self.optimizer,
            "loss_curve": list(self.loss_curve),
            "rate_curve": list(self.rate_curve),
            "validation_curve": list(self.validation_curve),
        }


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
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value


def snapshot_of(model):
    shape, weights = model.state()
    if isinstance(weights, dict):
        weights = {key: _clone(value) for key, value in weights.items()}
    else:
        weights = _clone(weights)
    return Snapshot(
        kind=model.kind,
        merges=model.tokenizer.merges,
        shape=shape,
        weights=weights,
    )


def run_snapshot(model, run_state):
    snapshot = snapshot_of(model)
    return Snapshot(
        kind=snapshot.kind,
        merges=snapshot.merges,
        shape=snapshot.shape,
        weights=snapshot.weights,
        run=RunState(**run_state),
    )


def restore(snapshot: Snapshot):
    for cls in (Bigram, NeuralBigram, MlpLM, TransformerLM):
        if cls.kind == snapshot.kind:
            return cls.from_state(
                TokenizerEncoderDecoder(snapshot.merges),
                snapshot.shape,
                snapshot.weights,
            )
    raise ValueError(f"unknown checkpoint kind {snapshot.kind!r}")
