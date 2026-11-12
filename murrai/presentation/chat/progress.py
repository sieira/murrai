from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from murrai.application.ports import TrainingEvent


class ProgressLine:
    def __init__(self, console=None):
        self._out = console or Console()
        self._progress = None
        self._task = None

    def __call__(self, event: TrainingEvent) -> None:
        if self._progress is None:
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=self._out,
            )
            self._task = self._progress.add_task("training", total=event.total_steps)
            self._progress.start()
        self._progress.update(
            self._task,
            completed=event.step,
            total=event.total_steps,
        )

    def stop(self) -> None:
        if self._progress is None:
            return
        elapsed = self._progress.tasks[0].elapsed
        self._progress.stop()
        self._out.print(f"training finished in {elapsed:.0f}s", markup=False)
