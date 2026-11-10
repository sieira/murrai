from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import (
    die_fragments,
    interactive_chat,
    plain_text,
    run_chat,
)


class Collector:
    def __init__(self):
        self.chunks = []

    def __call__(self, fragments):
        self.chunks.append(plain_text(fragments))

    def last(self):
        return self.chunks[-1]

    def any(self, needle):
        return any(needle in chunk for chunk in self.chunks)


class FakeSession:
    def __init__(self):
        self.calls = []
        self._text = ""
        self.roll = "e"

    @property
    def text(self):
        return self._text

    def commit(self, new_text):
        self.calls.append(("commit", new_text))
        if "z" in new_text:
            raise ValueError("unknown char")
        delta = new_text[len(self._text) :]
        self._text = new_text
        return delta

    def dice(self):
        self.calls.append(("dice",))
        return [("e", 0.12), ("t", 0.09), ("o", 0.08)]

    def send(self):
        self.calls.append(("send",))
        self._text += self.roll
        return self.roll


def test_die_fragments_show_the_split():
    fragments = die_fragments([("e", 0.12), ("t", 0.09), ("o", 0.08)], rolled="e")
    text = plain_text(fragments)
    assert "e  12.0%" in text
    assert "t   9.0%" in text
    assert "→ e" in text


def test_die_fragments_mark_the_rolled_row():
    fragments = die_fragments([("e", 0.12), ("t", 0.09)], rolled="e")
    styles = [style for style, _ in fragments]
    assert "class:die-hit" in styles
    assert "class:rolled" in styles


def test_die_fragments_without_a_roll_preview():
    fragments = die_fragments([("e", 0.12), ("t", 0.09)], rolled=None)
    assert "class:rolled" not in [style for style, _ in fragments]


def test_die_fragments_show_the_tail_count():
    ten = [(chr(97 + i), 0.1) for i in range(10)]
    text = plain_text(die_fragments(ten, rolled=None))
    assert "... 2 more" in text


def test_run_chat_scripted():
    session = FakeSession()
    out = Collector()
    lines = iter(["Cr", None])
    run_chat(session, lambda story: next(lines), out)
    assert out.last().endswith("bye\n")
    assert ("commit", "Cr") in session.calls
    assert ("send",) in session.calls
    assert out.any("e  12.0%")
    assert out.any("→ e")


def test_run_chat_prefills_the_next_prompt_with_the_story():
    session = FakeSession()
    out = Collector()
    received = []
    lines = iter(["Cr", None])

    def read(story):
        received.append(story)
        return next(lines)

    run_chat(session, read, out)
    assert received == ["", "Cre"]


def test_run_chat_empty_buffer_clears_the_story_and_rerolls():
    session = FakeSession()
    out = Collector()
    lines = iter(["a", "", None])
    run_chat(session, lambda story: next(lines), out)
    assert out.last().endswith("bye\n")
    assert session.text == session.roll


def test_run_chat_ctrl_c_quits():
    session = FakeSession()
    out = Collector()

    def read(story):
        raise KeyboardInterrupt

    run_chat(session, read, out)
    assert out.last().endswith("bye\n")


def test_run_chat_unknown_char_message():
    session = FakeSession()
    out = Collector()
    lines = iter(["z", None])
    run_chat(session, lambda story: next(lines), out)
    assert out.any("unknown")
    assert session.text == ""


def test_run_chat_eof_read_returns_none():
    session = FakeSession()
    out = Collector()
    run_chat(session, lambda story: None, out)
    assert out.last().endswith("bye\n")


def test_the_real_prompt_reader_scripts_end_to_end():
    model = Bigram.fit("abab", Vocabulary.from_text("ab"))
    session = ChatSession(model, seed=0)
    with create_pipe_input() as pipe:
        pipe.send_text("ab\r")
        pipe.close()
        interactive_chat(session, input=pipe, output=DummyOutput())
    assert session.text.startswith("ab")
    assert len(session.text) >= 3
