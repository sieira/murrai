from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from murrai.application.checkpoints import CheckpointRecord, Snapshot
from murrai.application.sessions import StorySave
from murrai.domain.tokenizer import TokenizerEncoderDecoder


@dataclass(frozen=True)
class TrainingEvent:
    step: int
    total_steps: int
    phase: str = "training"


class EventBus(Protocol):
    def subscribe(self, listener: Callable[[TrainingEvent], None]) -> None: ...

    def unsubscribe(self, listener: Callable[[TrainingEvent], None]) -> None: ...

    def emit(self, event: TrainingEvent) -> None: ...


class Corpus(Protocol):
    def text(self) -> str: ...


@runtime_checkable
class LanguageModel(Protocol):
    kind: str
    tokenizer: TokenizerEncoderDecoder

    def probs(self, context_ids: list[int]): ...


class CheckpointStore(Protocol):
    def save(self, name: str, snapshot: Snapshot) -> None: ...

    def save_record(self, name: str, record: CheckpointRecord) -> None: ...

    def load(self, name: str) -> Snapshot: ...

    def record(self, name: str) -> CheckpointRecord: ...

    def names(self) -> list[str]: ...


class StoryStore(Protocol):
    def save(self, name: str, save: StorySave) -> None: ...

    def load(self, name: str) -> StorySave: ...

    def names(self) -> list[str]: ...
