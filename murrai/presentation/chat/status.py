import threading

BAR_WIDTH = 16


class TrainingStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[int, int] | None] = {}
        self.redraw = None

    def track(self, name):
        def on_event(event):
            with self._lock:
                self._runs[name] = (event.step, event.total_steps)
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

    def lines(self):
        with self._lock:
            return dict(self._runs)

    def has_lines(self):
        return bool(self.lines())

    def _wake(self):
        if self.redraw is not None:
            self.redraw()


def status_fragments(status):
    fragments = []
    for name, run in status.lines().items():
        if fragments:
            fragments.append(("class:hint", " │ "))
        if run is None:
            fragments.append(("class:failed", f"{name} failed"))
        else:
            step, total = run
            filled = BAR_WIDTH * step // total if total else BAR_WIDTH
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            fragments.append(("class:bar", f"{name} {bar} {step}/{total}"))
    return fragments


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
