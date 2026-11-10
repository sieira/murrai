from typing import Protocol, runtime_checkable

from murrai.application.checkpoints import Snapshot
from murrai.domain.vocabulary import Vocabulary


class DataSource(Protocol):
    def text(self) -> str: ...


@runtime_checkable
class LanguageModel(Protocol):
    vocabulary: Vocabulary

    def probs(self, context_ids: list[int]): ...
