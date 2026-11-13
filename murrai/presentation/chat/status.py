import contextlib
import multiprocessing
import os
import threading
import time

from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.lanes import (
    SILENT,
    TrainingFailed,
    TrainingFinished,
    TrainingStarted,
    TrainingStepped,
)

BAR_WIDTH = 16
WAKE_EVERY = 0.1


class TrainingStatus:
    def __init__(self, clock=time.monotonic, observer=SILENT):
        self._observer = observer
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[float, int, int] | None] = {}
        self.redraw = None
        self._clock = clock
        self._last_wake = float("-inf")

    def begin(self, name):
        with self._lock:
            self._runs[name] = (self._clock(), 0, 0)
        self._observer.on_event(TrainingStarted(name))
        self._wake(force=True)

    def track(self, name):
        def on_event(event):
            with self._lock:
                run = self._runs.get(name)
            if run is None:
                self.begin(name)
            with self._lock:
                start, _, total = self._runs[name]
                if total == 0:
                    start = self._clock()
                self._runs[name] = (start, event.step, event.total_steps)
            self._observer.on_event(
                TrainingStepped(name, event.step, event.total_steps),
            )
            self._wake()

        return on_event

    def finish(self, name):
        with self._lock:
            self._runs.pop(name, None)
        self._observer.on_event(TrainingFinished(name))
        self._wake(force=True)

    def fail(self, name, reason=""):
        with self._lock:
            self._runs[name] = None
        self._observer.on_event(TrainingFailed(name, reason))
        self._wake(force=True)

    def rows(self):
        with self._lock:
            items = list(self._runs.items())
        rows = []
        for name, run in items:
            if run is None:
                rows.append((name, None))
                continue
            start, step, total = run
            eta = None
            if step > 0 and total > 0:
                elapsed = self._clock() - start
                if elapsed > 0:
                    eta = (total - step) * elapsed / step
            rows.append((name, (step, total, eta)))
        return rows

    def _wake(self, force=False):
        now = self._clock()
        if not force and now - self._last_wake < WAKE_EVERY:
            return
        self._last_wake = now
        if self.redraw is not None:
            self.redraw()


def status_fragments(status):
    fragments = []
    for name, run in status.rows():
        if fragments:
            fragments.append(("class:hint", " │ "))
        if run is None:
            fragments.append(("class:failed", f"{name} failed"))
            continue
        step, total, eta = run
        if total == 0:
            fragments.append(("class:hint", f"{name} starting"))
            continue
        filled = BAR_WIDTH * step // total if total else BAR_WIDTH
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        text = f"{name} {bar} {step}/{total}"
        if eta is not None:
            text += f" eta {_clock_left(eta)}"
        fragments.append(("class:bar", text))
    return fragments


def _clock_left(seconds):
    seconds = int(seconds + 0.5)
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}:{s:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{s:02d}"


def _train_in_process(command, connection):
    def on_event(event):
        connection.send(("step", event.step, event.total_steps))

    try:
        built = command.build(on_event=on_event)
        connection.send(("done", built.kind if built is not None else None))
    except Exception as error:
        connection.send(("failed", str(error)))
    finally:
        connection.close()


def _pump(name, connection, status, on_built=None):
    on_event = status.track(name)
    while True:
        try:
            message = connection.recv()
        except EOFError:
            status.fail(name, "the training process ended without a result")
            return
        if message[0] == "step":
            on_event(TrainingEvent(step=message[1], total_steps=message[2]))
        elif message[0] == "done":
            built = message[1]
            if built is not None and on_built is not None:
                on_built(built)
            status.finish(name)
            return
        else:
            status.fail(name, message[1])
            return


# The child imports torch while unpickling, before any code of ours runs there;
# its warnings would print over the full-screen interface.
@contextlib.contextmanager
def _silenced_stdio():
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for descriptor in (*saved, devnull):
            os.close(descriptor)


def spawn_training(command, status, on_built=None):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_train_in_process,
        args=(command, send),
        daemon=True,
    )
    status.begin(command.name)
    try:
        with _silenced_stdio():
            process.start()
    except Exception as error:
        status.fail(command.name, str(error))
        return
    finally:
        send.close()
    threading.Thread(
        target=_pump,
        args=(command.name, receive, status, on_built),
        daemon=True,
    ).start()
