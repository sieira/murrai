from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Command:
    name: str
    description: str
    build: Callable
    trains: bool = False


@dataclass
class FormCommand:
    name: str
    description: str
    build: Callable
    fields: tuple
    bake: Callable
    trains: bool = False
    estimates: Callable | None = None
    continues: bool = False
    pin: dict | None = None
