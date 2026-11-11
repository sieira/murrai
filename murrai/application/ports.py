from typing import Protocol, runtime_checkable

from murrai.application.checkpoints import Snapshot
from murrai.domain.vocabulary import Vocabulary


class Corpus(Protocol):
    def text(self) -> str: ...


@runtime_checkable
class LanguageModel(Protocol):
    vocabulary: Vocabulary

    def probs(self, context_ids: list[int]): ...


class CheckpointStore(Protocol):
    def save(self, name: str, snapshot: Snapshot) -> None: ...

    def load(self, name: str) -> Snapshot: ...

    def names(self) -> list[str]: ...
