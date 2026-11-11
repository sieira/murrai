import threading
from collections import deque

from murrai.application.ports import TrainingEvent


class EtaLine:
    def __init__(self, write=print, interval=1.0):
        self._write = write
        self._interval = interval
        self._latest: deque[TrainingEvent] = deque(maxlen=1)
        self._rendered: TrainingEvent | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self, event: TrainingEvent) -> None:
        self._latest.append(event)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def drain(self) -> None:
        if self._latest and self._latest[0] is not self._rendered:
            self._rendered = self._latest[0]
            self._write(_eta_line(self._rendered))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.drain()


def _eta_line(event: TrainingEvent) -> str:
    remaining = event.elapsed_ms * (event.total_steps - event.step) / event.step
    done = event.step / event.total_steps
    return (
        f"step {event.step}/{event.total_steps} ({done:.0%})"
        f"  ETA {remaining / 1000:.0f}s"
    )
