from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Progress:
    step: int
    total_steps: int


class ProgressBus(Protocol):
    def emit(self, event: Progress) -> None: ...
