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
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._task = self._progress.add_task("training", total=None)

    def __call__(self, event: TrainingEvent) -> None:
        self._progress.update(
            self._task,
            completed=event.step,
            total=event.total_steps,
        )

    def start(self) -> None:
        self._progress.start()

    def stop(self) -> None:
        self._progress.stop()
