import io

from rich.console import Console

from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.progress import ProgressLine


def _console(file):
    return Console(file=file, width=80, force_terminal=True)


def test_the_progress_bar_reaches_the_final_step():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line.start()
    line(TrainingEvent(step=1999, total_steps=2000))
    line(TrainingEvent(step=2000, total_steps=2000))
    line.stop()

    text = file.getvalue()
    assert "2000/2000" in text
    assert "100%" in text


def test_the_progress_bar_shows_the_step_it_sits_on():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line.start()
    line(TrainingEvent(step=500, total_steps=2000))
    line.stop()

    assert "500/2000" in file.getvalue()


def test_the_bar_starts_empty_and_fills_with_events():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line(TrainingEvent(step=1, total_steps=2000))
    line.start()
    line.stop()

    assert "1/2000" in file.getvalue()
