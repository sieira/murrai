import io

from rich.console import Console

from murrai.application.ports import TrainingEvent
from murrai.presentation.chat.progress import ProgressLine


def _console(file):
    return Console(file=file, width=80, force_terminal=True)


def test_the_bar_starts_at_the_first_event_and_shows_counts():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line(TrainingEvent(step=1, total_steps=2000))
    line.stop()

    text = file.getvalue()
    assert "1/2000" in text
    assert "None" not in text


def test_the_bar_leaves_a_summary_instead_of_a_stuck_bar():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line(TrainingEvent(step=2000, total_steps=2000))
    line.stop()

    text = file.getvalue()
    assert "finished in" in text
    assert "100%" in text


def test_stopping_before_any_event_renders_nothing():
    file = io.StringIO()
    line = ProgressLine(console=_console(file))
    line.stop()

    assert file.getvalue() == ""
