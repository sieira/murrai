import math
import threading
import time
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Estimate:
    text: str
    runs: int


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


def gpt_cost(config):
    return (
        config["batch"]
        * config["max_time"]
        * config["depth"]
        * config["dimension"] ** 2
    )


def mlp_cost(config):
    return config["batch"] * config["window"] * config["dimension"] * config["hidden"]


def neural_cost(config):
    return config["batch"] * (256 + config["merges"])


def _guarded(cost, config):
    try:
        value = cost(config)
    except KeyError:
        return None
    return value if value > 0 else None


def _gradient_of(records, kind):
    for record in records:
        settings = record.settings
        gradient = record.gradient
        if settings is not None and settings.model == kind and gradient is not None:
            yield record, settings, gradient


def estimate_seconds(records, values, kind, cost):
    now_cost = _guarded(cost, values)
    if now_cost is None:
        return None
    best = None
    best_gap = math.inf
    runs = 0
    for _, _, gradient in _gradient_of(records, kind):
        past_cost = _guarded(cost, {**gradient.shape, "batch": gradient.batch})
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
        gradient.seconds * (values["steps"] / gradient.steps) * (now_cost / past_cost)
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
    def __init__(self, measure, cost):
        self._measure = measure
        self._cost = cost
        self._result = None
        self._kicked = False

    def kick(self):
        self._kicked = True
        threading.Thread(target=self.refresh, daemon=True).start()

    def refresh(self):
        per_step, config = self._measure()
        self._result = (per_step, config) if per_step else None

    @property
    def pending(self):
        return self._kicked and self._result is None

    def project(self, config):
        if self._result is None:
            return None
        per_step, past = self._result
        now_cost = _guarded(self._cost, config)
        past_cost = _guarded(self._cost, past)
        if now_cost is None or past_cost is None:
            return None
        return per_step * (now_cost / past_cost) * config["steps"]
