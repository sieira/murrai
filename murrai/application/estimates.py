import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Estimate:
    text: str
    runs: int


@dataclass(frozen=True)
class Cost:
    keys: frozenset
    of: Callable[[dict], float]


GPT_COST = Cost(
    keys=frozenset({"batch", "max_time", "depth", "dimension"}),
    of=lambda config: (
        config["batch"]
        * config["max_time"]
        * config["depth"]
        * config["dimension"] ** 2
    ),
)

MLP_COST = Cost(
    keys=frozenset({"batch", "window", "dimension", "hidden"}),
    of=lambda config: (
        config["batch"] * config["window"] * config["dimension"] * config["hidden"]
    ),
)

NEURAL_COST = Cost(
    keys=frozenset({"batch", "merges"}),
    of=lambda config: config["batch"] * (256 + config["merges"]),
)


def _cost_of(cost, config):
    if not cost.keys <= config.keys():
        return None
    return cost.of(config)


def format_seconds(seconds):
    seconds = max(round(seconds), 1)
    if seconds >= 3600:
        hours, rest = divmod(seconds, 3600)
        minutes = rest // 60
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    if seconds >= 60:
        minutes, rest = divmod(seconds, 60)
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    return f"{seconds} s"


def uniform_perplexity(merges):
    return 256 + merges


def _gradient_of(records, kind):
    for record in records:
        settings = record.settings
        gradient = record.gradient
        if settings is not None and settings.model == kind and gradient is not None:
            yield record, settings, gradient


def estimate_seconds(records, config, kind, cost):
    now_cost = _cost_of(cost, config)
    if now_cost is None:
        return None
    best = None
    best_gap = math.inf
    runs = 0
    for _, _, gradient in _gradient_of(records, kind):
        past_cost = _cost_of(cost, {**gradient.shape, "batch": gradient.batch})
        if past_cost is None or gradient.seconds is None or gradient.steps <= 0:
            continue
        runs += 1
        gap = abs(math.log(now_cost / past_cost))
        if gap < best_gap:
            best_gap = gap
            best = (gradient, past_cost)
    if best is None:
        return None
    gradient, past_cost = best
    seconds = (
        gradient.seconds * (config["steps"] / gradient.steps) * (now_cost / past_cost)
    )
    return Estimate(text=f"about {format_seconds(seconds)}", runs=runs)


def estimate_perplexity(records, kind, corpus):
    scores = [
        record.held_out_perplexity
        for record, settings, _ in _gradient_of(records, kind)
        if settings.corpus == corpus and record.held_out_perplexity is not None
    ]
    if not scores:
        return None
    if len(scores) == 1:
        return Estimate(
            text=f"the last run like this landed at ppl {scores[0]:.1f}",
            runs=1,
        )
    return Estimate(
        text=(
            f"runs like this landed between ppl {min(scores):.1f}"
            f" and {max(scores):.1f} (from {len(scores)} runs)"
        ),
        runs=len(scores),
    )


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


class Calibrator:
    # states: idle → running → ready | failed; a failed or ready
    # calibration stays put until begin() is asked for again.
    def __init__(self, measure, cost):
        self._measure = measure
        self._cost = cost
        self._state = "idle"
        self._result = None

    def begin(self):
        if self._state in ("idle", "failed"):
            self._state = "running"

    def refresh(self):
        if self._state != "running":
            return
        try:
            per_step, config = self._measure()
        except Exception:
            self._state = "failed"
            return
        self._result = (per_step, config) if per_step else None
        self._state = "ready" if self._result else "failed"

    @property
    def pending(self):
        return self._state == "running"

    def project(self, config):
        if self._state != "ready":
            return None
        now_cost = _cost_of(self._cost, config)
        past_cost = _cost_of(self._cost, self._result[1])
        if now_cost is None or past_cost is None:
            return None
        return self._result[0] * (now_cost / past_cost) * config["steps"]
