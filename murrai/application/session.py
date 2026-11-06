import torch

from murrai.domain.bigram import Bigram


class ChatSession:
    def __init__(self, model: Bigram, seed: int):
        self._model = model
        self._generator = torch.Generator().manual_seed(seed)

    def _char_to_id(self, char):
        (char_id,) = self._model.vocabulary.encode(char)
        return char_id

    def dice(self, char: str):
        row = self._model.probs(self._char_to_id(char))
        dice = list(zip(self._model.vocabulary.chars, row.tolist()))
        return sorted(dice, key=lambda entry: entry[1], reverse=True)

    def roll(self, char: str) -> str:
        char_id = self._model.roll(self._char_to_id(char), self._generator)
        return self._model.vocabulary.chars[char_id]
