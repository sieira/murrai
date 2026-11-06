import random

from murrai.application.ports import LanguageModel


class ChatSession:
    def __init__(self, model: LanguageModel, seed: int):
        self._model = model
        self._rng = random.Random(seed)

    def _char_to_id(self, char):
        (char_id,) = self._model.vocabulary.encode(char)
        return char_id

    def dice(self, char: str):
        row = [float(p) for p in self._model.probs(self._char_to_id(char))]
        dice = list(zip(self._model.vocabulary.chars, row))
        return sorted(dice, key=lambda entry: entry[1], reverse=True)

    def roll(self, char: str) -> str:
        row = [float(p) for p in self._model.probs(self._char_to_id(char))]
        r = self._rng.random() * sum(row)
        so_far = 0.0
        for index, p in enumerate(row):
            so_far += p
            if r < so_far:
                return self._model.vocabulary.chars[index]
        return self._model.vocabulary.chars[-1]
