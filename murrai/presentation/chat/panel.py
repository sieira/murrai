import textwrap
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Command:
    name: str
    description: str
    build: Callable
    trains: bool = False


class CommandPanel:
    def __init__(self, commands):
        self._commands = commands
        self.menu = []
        self.selected = 0
        self.visible = False

    @property
    def current(self):
        return self.menu[self.selected] if self.menu else None

    def open(self):
        self.menu = list(
            self._commands() if callable(self._commands) else self._commands,
        )
        self.selected = 0
        self.visible = True

    def close(self):
        self.visible = False

    def move(self, step):
        if self.menu:
            self.selected = (self.selected + step) % len(self.menu)


def list_fragments(panel):
    fragments = []
    for i, command in enumerate(panel.menu):
        style = "class:selected" if i == panel.selected else ""
        fragments.append((style, command.name))
        fragments.append(("", "\n"))
    return fragments


def description_fragments(panel, width=None):
    command = panel.current
    if command is None:
        return []
    text = textwrap.fill(command.description, width) if width else command.description
    return [("class:hint", text)]
