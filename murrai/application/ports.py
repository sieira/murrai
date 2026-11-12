from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from murrai.application.checkpoints import Snapshot
from murrai.domain.vocabulary import Vocabulary


@dataclass(frozen=True)
class TrainingEvent:
    step: int
    total_steps: int


class EventBus(Protocol):
    def subscribe(self, listener: Callable[[TrainingEvent], None]) -> None: ...

    def unsubscribe(self, listener: Callable[[TrainingEvent], None]) -> None: ...

    def emit(self, event: TrainingEvent) -> None: ...


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
