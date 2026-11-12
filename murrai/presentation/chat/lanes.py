import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatEvent:
    def lane_line(self) -> str:
        return f"[event] {self}"


@dataclass(frozen=True)
class PanelOpened(ChatEvent):
    count: int

    def lane_line(self):
        return f"[panel] opened {self.count} commands"


@dataclass(frozen=True)
class PanelClosed(ChatEvent):
    def lane_line(self):
        return "[panel] closed"


@dataclass(frozen=True)
class CommandPicked(ChatEvent):
    name: str

    def lane_line(self):
        return f"[command] picked {self.name}"


@dataclass(frozen=True)
class SessionSwapped(ChatEvent):
    def lane_line(self):
        return "[session] swapped"


@dataclass(frozen=True)
class TrainingStarted(ChatEvent):
    name: str

    def lane_line(self):
        return f"[train] {self.name} started"


@dataclass(frozen=True)
class TrainingStepped(ChatEvent):
    name: str
    step: int
    total: int

    def lane_line(self):
        return f"[train] {self.name} {self.step}/{self.total}"


@dataclass(frozen=True)
class TrainingFinished(ChatEvent):
    name: str

    def lane_line(self):
        return f"[train] {self.name} finished"


@dataclass(frozen=True)
class TrainingFailed(ChatEvent):
    name: str
    reason: str = ""

    def lane_line(self):
        return f"[train] {self.name} failed: {self.reason}"


class EventObserver:
    def on_event(self, event: ChatEvent) -> None:
        pass


class ChatLog(EventObserver):
    def __init__(self, path=None, clock=time.monotonic):
        self.lines: list[str] = []
        self._path = path
        self._clock = clock

    def on_event(self, event):
        line = f"{self._clock():.3f} {event.lane_line()}"
        self.lines.append(line)
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as log:
                log.write(line + "\n")
