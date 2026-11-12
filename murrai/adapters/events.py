import threading

from murrai.application.ports import TrainingEvent


class ListenerBus:
    def __init__(self):
        self._listeners: list = []
        self._lock = threading.Lock()

    def subscribe(self, listener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners = [*self._listeners, listener]

    def unsubscribe(self, listener) -> None:
        with self._lock:
            self._listeners = [known for known in self._listeners if known != listener]

    def emit(self, event: TrainingEvent) -> None:
        for listener in self._listeners:
            listener(event)
