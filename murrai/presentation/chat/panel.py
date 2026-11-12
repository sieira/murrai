from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Command:
    name: str
    description: str
    build: Callable
    trains: bool = False


def filter_commands(commands, query):
    query = query.strip().lower()
    if not query:
        return list(commands)
    return [
        command
        for command in commands
        if query in command.name.lower() or query in command.description.lower()
    ]
