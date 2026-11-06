from typing import Protocol


class DataSource(Protocol):
    def text(self) -> str: ...
