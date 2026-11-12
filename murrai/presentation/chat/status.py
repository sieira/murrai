import threading
import time

BAR_WIDTH = 16


class TrainingStatus:
    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[float, int, int] | None] = {}
        self.redraw = None
        self._clock = clock

    def track(self, name):
        def on_event(event):
            with self._lock:
                if name not in self._runs or self._runs[name] is None:
                    start = self._clock()
                else:
                    start = self._runs[name][0]
                self._runs[name] = (start, event.step, event.total_steps)
            self._wake()

        return on_event

    def finish(self, name):
        with self._lock:
            self._runs.pop(name, None)
        self._wake()

    def fail(self, name):
        with self._lock:
            self._runs[name] = None
        self._wake()

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

    def has_lines(self):
        return bool(self.rows())

    def _wake(self):
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


def _train(command, status):
    listener = status.track(command.name)
    try:
        command.build(on_event=listener)
    except Exception:
        status.fail(command.name)
    else:
        status.finish(command.name)


def spawn_training(command, status):
    threading.Thread(target=_train, args=(command, status), daemon=True).start()
