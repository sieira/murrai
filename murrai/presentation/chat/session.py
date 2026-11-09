import random
from dataclasses import dataclass

from murrai.application.ports import LanguageModel


@dataclass
class Turn:
    dice: list[tuple[str, float]]
    rolled: str
    history: str


class ChatSession:
    def __init__(self, model: LanguageModel, seed: int):
        self._model = model
        self._rng = random.Random(seed)
        self._ids: list[int] = []

    def turn(self, char: str) -> Turn:
        (char_id,) = self._model.vocabulary.encode(char)
        self._ids.append(char_id)
        row = [float(p) for p in self._model.probs(list(self._ids))]
        dice = sorted(
            zip(self._model.vocabulary.chars, row),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return Turn(
            dice=dice,
            rolled=self._sample(row),
            history=self._model.vocabulary.decode(self._ids),
        )

    def _sample(self, row):
        r = self._rng.random() * sum(row)
        so_far = 0.0
        for index, p in enumerate(row):
            so_far += p
            if r < so_far:
                return self._model.vocabulary.chars[index]
        return self._model.vocabulary.chars[-1]
