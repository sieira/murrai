import itertools
import random

from murrai.application.ports import LanguageModel


class ChatSession:
    def __init__(self, model: LanguageModel, seed: int):
        self._model = model
        self._rng = random.Random(seed)
        self._ids: list[int] = []
        self._origins: list[str] = []

    @property
    def kind(self) -> str:
        return self._model.kind

    @property
    def chars(self) -> tuple:
        return self._model.vocabulary.chars

    @property
    def text(self) -> str:
        return self._model.vocabulary.decode(self._ids)

    def truncate(self, keep: int) -> None:
        self._ids = self._ids[:keep]
        self._origins = self._origins[:keep]

    def extend_user(self, chars: str) -> None:
        self._extend(chars, "entered")

    def extend_generated(self, chars: str) -> None:
        self._extend(chars, "generated")

    def _extend(self, chars: str, origin: str) -> None:
        fresh = list(self._model.vocabulary.encode(chars))
        self._ids.extend(fresh)
        self._origins.extend([origin] * len(fresh))

    def grow(self, chars: str) -> None:
        grown = self._model.vocabulary.grown_with(chars)
        if grown.chars == self._model.vocabulary.chars:
            return
        self._model = self._model.grown_to(grown)

    def dice(self):
        return self._dice(self._ids)

    def preview(self, text: str):
        return self._dice(list(self._model.vocabulary.encode(text)))

    def _dice(self, ids):
        row = [float(p) for p in self._model.probs(list(ids))]
        pairs = sorted(
            zip(self._model.vocabulary.chars, row, strict=False),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return pairs

    def rollout(self, text: str, n: int) -> str:
        ids = list(self._model.vocabulary.encode(text))
        chars = []
        for _ in range(n):
            row = [float(p) for p in self._model.probs(list(ids))]
            index = self._sample(row)
            ids.append(index)
            chars.append(self._model.vocabulary.chars[index])
        return "".join(chars)

    def _sample(self, row):
        r = self._rng.random() * sum(row)
        so_far = 0.0
        for index, p in enumerate(row):
            so_far += p
            if r < so_far:
                return index
        return len(row) - 1

    def segments(self):
        segments = []
        position = 0
        for origin, run in itertools.groupby(self._origins):
            count = len(list(run))
            segment = self._model.vocabulary.decode(
                self._ids[position : position + count],
            )
            segments.append((segment, origin))
            position += count
        return segments
