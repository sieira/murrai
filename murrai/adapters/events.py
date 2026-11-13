import queue
import threading

from murrai.application.ports import TrainingEvent


class ListenerBus:
    # Emitting hands the event to a queue and returns; a worker thread
    # delivers. A listener that is slow or raises can therefore never slow
    # or stop the code that emits (ADR-0005).
    def __init__(self):
        self._listeners: list = []
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self.listener_errors: list = []
        self._worker = threading.Thread(target=self._deliver, daemon=True)
        self._worker.start()

    def subscribe(self, listener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners = [*self._listeners, listener]

    def unsubscribe(self, listener) -> None:
        with self._lock:
            self._listeners = [known for known in self._listeners if known != listener]

    def emit(self, event: TrainingEvent) -> None:
        self._queue.put(event)

    def flush(self) -> None:
        self._queue.join()

    def _deliver(self) -> None:
        while True:
            event = self._queue.get()
            with self._lock:
                listeners = [*self._listeners]
            for listener in listeners:
                try:
                    listener(event)
                except Exception as error:
                    self.listener_errors.append(error)
            self._queue.task_done()
