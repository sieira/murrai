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
    def text(self) -> str:
        return self._model.tokenizer.decode(self._ids)

    @property
    def labels(self):
        return self._model.tokenizer.labels

    def truncate(self, keep: int) -> None:
        self._ids = self._ids[:keep]
        self._origins = self._origins[:keep]

    def extend_user(self, text: str) -> None:
        self._extend(text, "entered")

    def extend_generated(self, text: str) -> None:
        self._extend(text, "generated")

    def _extend(self, text: str, origin: str) -> None:
        fresh = list(self._model.tokenizer.encode(text))
        self._ids.extend(fresh)
        self._origins.extend([origin] * len(fresh))

    def dice(self):
        return self._dice(self._ids)

    def preview(self, text: str):
        return self._dice(list(self._model.tokenizer.encode(text)))

    def _dice(self, ids):
        row = [float(p) for p in self._model.probs(list(ids))]
        pairs = sorted(
            zip(self._model.tokenizer.labels, row, strict=True),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return pairs

    def rollout(self, text: str, n: int) -> str:
        return self.rollout_ids(text, n)[1]

    def rollout_ids(self, text: str, n: int) -> tuple[list[int], str]:
        ids = list(self._model.tokenizer.encode(text))
        pieces = self._model.tokenizer.pieces
        out = []
        for _ in range(n):
            row = [float(p) for p in self._model.probs(list(ids))]
            index = self._sample(row)
            ids.append(index)
            out.append(pieces[index])
        return ids, "".join(out)

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
            segment = self._model.tokenizer.decode(
                self._ids[position : position + count],
            )
            segments.append((segment, origin))
            position += count
        return segments
