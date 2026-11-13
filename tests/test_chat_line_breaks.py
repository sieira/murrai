import threading
import time

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.runs import train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import GhostChat, plain_text, run_chat


def _newline_session():
    model = train_neural_bigram("ab\nab", steps=3, batch=4, seed=0).model
    return ChatSession(model, 0)


def test_shift_enter_breaks_the_line_and_enter_still_commits():
    session = _newline_session()

    with create_pipe_input() as pipe:
        pipe.send_text("ab\x1b[13;2uab\r\x03")
        returned = run_chat(session, input=pipe, output=DummyOutput())

    assert returned is session
    assert session.text == "ab\nab"
    assert session.segments() == [("ab\nab", "entered")]


def test_alt_enter_breaks_the_line_and_enter_still_commits():
    session = _newline_session()

    with create_pipe_input() as pipe:
        pipe.send_text("ab\x1b\rab\r\x03")
        returned = run_chat(session, input=pipe, output=DummyOutput())

    assert returned is session
    assert session.text == "ab\nab"
    assert session.segments() == [("ab\nab", "entered")]


def _newline_heavy_chat():
    model = Bigram.fit("a\na\n", Vocabulary.from_text("a\n"))
    return GhostChat(ChatSession(model, 0))


def test_the_die_bar_shows_a_newline_row_as_the_break_mark():
    chat = _newline_heavy_chat()

    text = plain_text(chat.die_bar("a"))

    assert "↵" in text
    assert "\n" not in text


def test_escape_answers_quickly_even_with_the_chord_sequences():
    from murrai.presentation.chat.lanes import ChatLog

    log = ChatLog()
    with create_pipe_input() as pipe:

        def feed():
            time.sleep(0.3)
            pipe.send_text("\x10")
            time.sleep(0.1)
            pipe.send_text("\x1b")
            time.sleep(1.2)
            pipe.send_text("\x03")

        thread = threading.Thread(target=feed)
        thread.start()
        run_chat(_newline_session(), input=pipe, output=DummyOutput(), observer=log)
        thread.join(timeout=10)

    stamps = {}
    for line in log.lines:
        stamp, text = line.split(" ", 1)
        if text.startswith("[panel] opened"):
            stamps.setdefault("opened", float(stamp))
        if text == "[panel] closed":
            stamps.setdefault("closed", float(stamp))
    assert "closed" in stamps
    assert stamps["closed"] - stamps["opened"] < 0.5
