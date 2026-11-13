import textwrap
from collections.abc import Callable
from dataclasses import dataclass

from murrai.presentation.chat.lanes import SILENT, PanelClosed, PanelOpened


@dataclass
class Command:
    name: str
    description: str
    build: Callable
    trains: bool = False
    fields: tuple = ()
    bake: Callable | None = None
    given: dict | None = None


class CommandPanel:
    def __init__(self, commands, observer=SILENT):
        self._commands = commands
        self._observer = observer
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
        self._observer.on_event(PanelOpened(len(self.menu)))

    def close(self):
        self.visible = False
        self._observer.on_event(PanelClosed())

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
