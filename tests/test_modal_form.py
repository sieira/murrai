import threading
import time

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.containers import FloatContainer
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import Dialog

from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession, Command, run_chat
from murrai.presentation.chat.form import Field, FormOverlay, TrainingForm
from murrai.presentation.chat.panel import CommandPanel
from murrai.presentation.chat.repl import GhostChat, chat_layout
from murrai.presentation.chat.status import TrainingStatus


def _story_session(text="hello"):
    session = ChatSession(
        Bigram.fit(text, Vocabulary.from_text(text)),
        0,
    )
    session.extend_user(text)
    return session


def _form_command(fields=None):
    def bake(values):
        return Command(
            "retrain x",
            "knobs",
            lambda: None,
            trains=True,
            given=values,
        )

    return Command(
        "retrain x",
        "knobs",
        lambda: None,
        trains=True,
        fields=fields
        or (
            Field("steps", "steps", "int", 2000, check="whole above zero"),
            Field(
                "final_learning_rate",
                "final learning rate",
                "float",
                0.1,
                check="number above zero",
            ),
        ),
        bake=bake,
    )


def _layout_parts(command):
    panel = CommandPanel([command])
    focused = []
    overlay = FormOverlay(
        on_accept=lambda form, values: None,
        on_cancel=lambda: None,
        focus=focused.append,
    )
    layout = chat_layout(
        story=Window(),
        editor=Window(),
        die=Window(),
        panel_list=Window(),
        panel=panel,
        overlay=overlay,
        status=TrainingStatus(),
        chat=GhostChat(None),
    )
    return layout, overlay, focused


def test_the_form_is_a_float_over_the_chat_and_not_a_column():
    layout, overlay, focused = _layout_parts(_form_command())

    root = layout.container
    assert isinstance(root, FloatContainer)
    assert [root.floats] == [[overlay.float]]

    rows = root.content.get_children()
    assert len(rows) == 2
    columns = rows[0].get_children()
    assert len(columns) == 2

    float_content = root.floats[0].content
    assert float_content is overlay.container
    assert not float_content.filter()

    overlay.open(TrainingForm(_form_command()))
    assert isinstance(overlay.body, Dialog)
    assert float_content.filter()
    assert focused
    assert root.floats[0].top == 2
    assert root.floats[0].width is None
    assert root.floats[0].height is None
    assert root.floats[0].left is None
    assert root.floats[0].right is None

    overlay.close()
    assert not float_content.filter()


class _ScreenOutput(DummyOutput):
    def __init__(self):
        self.chunks = []
        self.snapshots = []
        self._calls = 0

    def get_size(self):
        self._calls += 1
        return Size(rows=40, columns=80 + self._calls)

    def write(self, data):
        self.chunks.append(data)

    def flush(self):
        if self.chunks:
            self.snapshots.append("".join(self.chunks))
            self.chunks = []


def _screens(output):
    return [
        snapshot.split("\r\n") for snapshot in output.snapshots if len(snapshot) > 1000
    ]


def _text(rows):
    return "\n".join(row.replace("\r", "") for row in rows)


def _chat_screen(keys):
    output = _ScreenOutput()
    with create_pipe_input() as pipe:
        box = {}
        failures = []

        def chat():
            try:
                box["session"] = run_chat(
                    _story_session(),
                    commands=[_form_command()],
                    input=pipe,
                    output=output,
                    load=lambda name: _story_session(),
                )
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=chat)
        thread.start()
        time.sleep(0.3)
        for text, delay in keys:
            pipe.send_text(text)
            time.sleep(delay)
        pipe.send_text("\x03")
        thread.join(timeout=10)
    if failures:
        raise failures[0]
    assert "session" in box
    return output


def test_the_open_form_floats_a_titled_box_over_the_story():
    output = _chat_screen([("\x10", 0.3), ("\r", 0.3)])

    rows = _screens(output)[-1]
    screen = _text(rows)
    assert "retrain x" in screen
    assert "steps:" in screen
    assert "final learning rate:" in screen
    assert "2000" in screen
    assert "hello" in screen

    title_row = next(row for row in rows if "retrain x" in row)
    assert title_row.index("retrain x") >= 4
    assert rows.index(title_row) >= 2
    assert rows[0].count("┌") == 1


def test_closing_the_form_leaves_no_box_behind():
    # a lone escape is delivered only after the parser flush (0.5s) plus the
    # key processor timeout (1s) have both passed
    output = _chat_screen([("\x10", 0.3), ("\r", 0.3), ("\x1b", 1.6)])

    screens = _screens(output)
    assert any("retrain x" in _text(rows) for rows in screens)
    rows = screens[-1]
    screen = _text(rows)
    assert "retrain x" not in screen
    assert "steps:" not in screen
    assert "final learning rate:" not in screen
    assert "hello" in screen


def test_the_modal_dresses_in_the_chat_colors_instead_of_the_library_white():
    from prompt_toolkit.styles import merge_styles
    from prompt_toolkit.styles.defaults import default_ui_style

    from murrai.presentation.chat.repl import STYLE

    merged = merge_styles([default_ui_style(), STYLE])
    body = merged.get_attrs_for_style_str("class:dialog.body")
    field = merged.get_attrs_for_style_str("class:dialog.body class:text-area")

    assert body.bgcolor == "default"
    assert body.color == "default"
    assert field.bgcolor == "default"
