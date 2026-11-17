import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from murrai.application.checkpoints import Shape


@dataclass(frozen=True)
class Cost:
    kind: str
    of: Callable[[Shape, int], float]


GPT_COST = Cost(
    kind="gpt",
    of=lambda shape, batch: batch * shape.max_time * shape.depth * shape.dimension**2,
)

MLP_COST = Cost(
    kind="windowed-mlp",
    of=lambda shape, batch: batch * shape.window * shape.dimension * shape.hidden,
)

NEURAL_COST = Cost(
    kind="neural-bigram",
    of=lambda shape, batch: batch * (256 + shape.merges),
)


def format_seconds(seconds):
    seconds = max(round(seconds), 1)
    if seconds >= 3600:
        hours, rest = divmod(seconds, 3600)
        return f"{hours} h {minutes} min" if (minutes := rest // 60) else f"{hours} h"
    if seconds >= 60:
        minutes, rest = divmod(seconds, 60)
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    return f"{seconds} s"


def _gradient_of(records, kind):
    for record in records:
        settings = record.settings
        gradient = record.gradient
        if settings is not None and settings.model == kind and gradient is not None:
            yield gradient


def estimate_seconds(records, shape, batch, steps, cost):
    now_cost = cost.of(shape, batch)
    best = None
    best_gap = math.inf
    for gradient in _gradient_of(records, cost.kind):
        if gradient.shape is None or gradient.seconds is None or gradient.steps <= 0:
            continue
        past_cost = cost.of(gradient.shape, gradient.batch)
        gap = abs(math.log(now_cost / past_cost))
        if gap < best_gap:
            best_gap = gap
            best = (gradient, past_cost)
    if best is None:
        return None
    gradient, past_cost = best
    scale = (steps / gradient.steps) * (now_cost / past_cost)
    return gradient.seconds * scale


class StepClock:
    def __init__(self, now=None):
        self._now = now
        self._stamps = []

    def emit(self, event):
        if event.phase != "training":
            return
        self._stamps.append(self._now() if self._now is not None else time.monotonic())

    def per_step(self):
        deltas = [later - earlier for earlier, later in pairwise(self._stamps)]
        kept = deltas[2:]
        if not kept:
            return None
        return sum(kept) / len(kept)


class CalibratorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


class Calibrator:
    # states: idle → running → ready | failed; a calibration is
    # measured once, then it stays put.
    def __init__(self, measure, cost):
        self._measure = measure
        self._cost = cost
        self._state = CalibratorState.IDLE
        self._result = None

    def begin(self):
        if self._state is CalibratorState.IDLE:
            self._state = CalibratorState.RUNNING

    def refresh(self):
        if self._state is not CalibratorState.RUNNING:
            return
        try:
            per_step, shape, batch = self._measure()
        except Exception:
            self._state = CalibratorState.FAILED
            return
        self._result = (per_step, shape, batch) if per_step else None
        self._state = CalibratorState.READY if self._result else CalibratorState.FAILED

    @property
    def pending(self):
        return self._state is CalibratorState.RUNNING

    def project(self, shape, batch, steps):
        if self._state is not CalibratorState.READY:
            return None
        now_cost = self._cost.of(shape, batch)
        past_cost = self._cost.of(self._result[1], self._result[2])
        return self._result[0] * (now_cost / past_cost) * steps
