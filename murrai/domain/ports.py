from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Progress:
    step: int
    total_steps: int


# The bus can only receive progress; it cannot be subscribed to. Whoever
# hands it in decides what a step means and who listens.
class ProgressBus(Protocol):
    def emit(self, event: Progress) -> None: ...
