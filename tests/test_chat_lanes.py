from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.ports import TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat import ChatSession, Command, run_chat
from murrai.presentation.chat.lanes import (
    ChatLog,
    CommandPicked,
    FormCancelled,
    FormOpened,
    SessionSwapped,
    TrainingFailed,
    TrainingFinished,
    TrainingStarted,
    TrainingStepped,
)
from murrai.presentation.chat.status import TrainingStatus


def _session(text="abab"):
    return ChatSession(Bigram.fit(text, TokenizerEncoderDecoder()), 0)


def test_each_event_renders_its_own_lane_line():
    assert CommandPicked("neural bigram").lane_line() == (
        "[command] picked neural bigram"
    )
    assert SessionSwapped().lane_line() == "[session] swapped"
    assert TrainingStarted("x").lane_line() == "[train] x started"
    assert TrainingStepped("x", 3, 50).lane_line() == "[train] x training 3/50"
    assert TrainingStepped("x", 3, 50, "tokenising").lane_line() == (
        "[train] x tokenising 3/50"
    )
    assert TrainingFinished("x").lane_line() == "[train] x finished"
    assert TrainingFailed("x", "boom").lane_line() == "[train] x failed: boom"
    assert FormOpened("retrain char gpt").lane_line() == (
        "[form] opened retrain char gpt"
    )
    assert FormCancelled("retrain char gpt").lane_line() == (
        "[form] cancelled retrain char gpt"
    )


def test_the_log_stamps_each_lane_line_with_the_clock():
    ticks = iter([1.5, 2.25])
    log = ChatLog(clock=lambda: next(ticks))

    log.on_event(CommandPicked("counted bigram"))
    log.on_event(SessionSwapped())

    assert log.lines == [
        "1.500 [command] picked counted bigram",
        "2.250 [session] swapped",
    ]


def test_the_log_appends_its_lines_to_a_file(tmp_path):
    path = tmp_path / "chat.log"
    log = ChatLog(path, clock=lambda: 0.0)

    log.on_event(CommandPicked("counted bigram"))
    log.on_event(SessionSwapped())

    assert path.read_text().splitlines() == [
        "0.000 [command] picked counted bigram",
        "0.000 [session] swapped",
    ]


def test_a_training_run_publishes_start_steps_and_finish():
    log = ChatLog(clock=lambda: 0.0)
    status = TrainingStatus(observer=log)

    on_event = status.track("x")
    on_event(TrainingEvent(step=1, total_steps=2))
    on_event(TrainingEvent(step=2, total_steps=2))
    status.finish("x")

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[train] x started",
        "[train] x training 1/2",
        "[train] x training 2/2",
        "[train] x finished",
    ]


def test_a_failed_training_run_publishes_why():
    log = ChatLog(clock=lambda: 0.0)
    status = TrainingStatus(observer=log)

    status.fail("x", "boom")

    assert log.lines == ["0.000 [train] x failed: boom"]


def test_picking_a_command_publishes_the_pick_and_the_swap():
    log = ChatLog(clock=lambda: 0.0)
    commands = (Command("counted bigram", "x", lambda: _session("baba")),)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r\x03")
        run_chat(
            _session(),
            menu=lambda: {"new": commands, "load": (), "retrain": (), "sessions": ()},
            input=pipe,
            output=DummyOutput(),
            observer=log,
        )

    assert [line.split(" ", 1)[1] for line in log.lines] == [
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
        "[train] x training 1/2",
    ]
