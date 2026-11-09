from typing import Protocol

from murrai.domain.vocabulary import Vocabulary


class DataSource(Protocol):
    def text(self) -> str: ...


class LanguageModel(Protocol):
    vocabulary: Vocabulary

    def probs(self, prev_id: int): ...
