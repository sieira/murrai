from dataclasses import dataclass


@dataclass(frozen=True)
class StorySave:
    ids: tuple[int, ...]
    origins: tuple[str, ...]
    checkpoint: str
