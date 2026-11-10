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
    def text(self) -> str:
        return self._model.vocabulary.decode(self._ids)

    def commit(self, new_text: str) -> str:
        old = self.text
        common = 0
        for a, b in zip(old, new_text):
            if a != b:
                break
            common += 1
        fresh = list(self._model.vocabulary.encode(new_text[common:]))
        self._ids = self._ids[:common] + fresh
        self._origins = self._origins[:common] + ["entered"] * len(fresh)
        return new_text[common:]

    def dice(self):
        return self._dice(self._ids)

    def preview(self, text: str):
        return self._dice(list(self._model.vocabulary.encode(text)))

    def _dice(self, ids):
        row = [float(p) for p in self._model.probs(list(ids))]
        pairs = sorted(
            zip(self._model.vocabulary.chars, row),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return pairs

    def send(self) -> str:
        row = [float(p) for p in self._model.probs(list(self._ids))]
        index = self._sample(row)
        self._ids.append(index)
        self._origins.append("generated")
        return self._model.vocabulary.chars[index]

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
                self._ids[position : position + count]
            )
            segments.append((segment, origin))
            position += count
        return segments
