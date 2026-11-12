from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.ports import TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession, Command, run_chat
from murrai.presentation.chat.lanes import (
    ChatLog,
    CommandPicked,
    PanelClosed,
    PanelOpened,
    SessionSwapped,
    TrainingFailed,
    TrainingFinished,
    TrainingStarted,
    TrainingStepped,
)
from murrai.presentation.chat.panel import CommandPanel
from murrai.presentation.chat.status import TrainingStatus


def _session(text="abab"):
    return ChatSession(Bigram.fit(text, Vocabulary.from_text("ab")), 0)


def test_each_event_renders_its_own_lane_line():
    assert PanelOpened(3).lane_line() == "[panel] opened 3 commands"
    assert PanelClosed().lane_line() == "[panel] closed"
    assert CommandPicked("neural bigram").lane_line() == (
        "[command] picked neural bigram"
    )
    assert SessionSwapped().lane_line() == "[session] swapped"
    assert TrainingStarted("x").lane_line() == "[train] x started"
    assert TrainingStepped("x", 3, 50).lane_line() == "[train] x 3/50"
    assert TrainingFinished("x").lane_line() == "[train] x finished"
    assert TrainingFailed("x", "boom").lane_line() == "[train] x failed: boom"


def test_the_log_stamps_each_lane_line_with_the_clock():
    ticks = iter([1.5, 2.25])
    log = ChatLog(clock=lambda: next(ticks))

    log.on_event(PanelClosed())
    log.on_event(SessionSwapped())

    assert log.lines == ["1.500 [panel] closed", "2.250 [session] swapped"]


def test_the_log_appends_its_lines_to_a_file(tmp_path):
    path = tmp_path / "chat.log"
    log = ChatLog(path, clock=lambda: 0.0)

    log.on_event(PanelClosed())
    log.on_event(SessionSwapped())

    assert path.read_text().splitlines() == [
        "0.000 [panel] closed",
        "0.000 [session] swapped",
    ]


def test_the_panel_publishes_when_it_opens_and_closes():
    log = ChatLog(clock=lambda: 0.0)
    panel = CommandPanel(
        [Command("a", "x", lambda: None), Command("b", "y", lambda: None)],
        observer=log,
    )

    panel.open()
    panel.close()

    assert log.lines == ["0.000 [panel] opened 2 commands", "0.000 [panel] closed"]


def test_a_training_run_publishes_start_steps_and_finish():
    log = ChatLog(clock=lambda: 0.0)
    status = TrainingStatus(observer=log)

    on_event = status.track("x")
    on_event(TrainingEvent(step=1, total_steps=2))
    on_event(TrainingEvent(step=2, total_steps=2))
    status.finish("x")

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[train] x started",
        "[train] x 1/2",
        "[train] x 2/2",
        "[train] x finished",
    ]


def test_a_failed_training_run_publishes_why():
    log = ChatLog(clock=lambda: 0.0)
    status = TrainingStatus(observer=log)

    status.fail("x", "boom")

    assert log.lines == ["0.000 [train] x failed: boom"]


def test_picking_a_command_publishes_the_pick_and_the_swap():
    log = ChatLog(clock=lambda: 0.0)
    commands = [Command("counted bigram", "x", lambda: _session("baba"))]

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r\x03")
        run_chat(
            _session(),
            commands=commands,
            input=pipe,
            output=DummyOutput(),
            observer=log,
        )

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[panel] opened 1 commands",
        "[panel] closed",
        "[command] picked counted bigram",
        "[session] swapped",
    ]


def test_beginning_a_run_publishes_started_once():
    log = ChatLog(clock=lambda: 0.0)
    status = TrainingStatus(observer=log)

    status.begin("x")
    status.track("x")(TrainingEvent(step=1, total_steps=2))

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[train] x started",
        "[train] x 1/2",
    ]
