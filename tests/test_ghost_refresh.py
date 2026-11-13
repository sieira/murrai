import threading
import time

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import (
    GHOST_CHARS,
    GhostChat,
    GhostSuggestions,
    run_chat,
)


def _session():
    return ChatSession(Bigram.fit("abab", Vocabulary.from_text("ab")), 0)


def test_a_second_tab_accept_rolls_fresh_shadow_text():
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\x03")
        run_chat(session, input=pipe, output=DummyOutput())

    assert len(session.text) == 2 * GHOST_CHARS
    assert [origin for _, origin in session.segments()] == ["generated"]


def test_after_an_accept_the_next_roll_is_fresh_not_a_cached_remainder():
    chat = GhostChat(_session())
    suggestions = GhostSuggestions(chat)
    first = suggestions.real_for("")
    chat.accept_rest("", first)
    suggestions.forget()
    second = suggestions.real_for(chat.session.text)

    suffixes = {first[k:] for k in range(len(first) + 1)}
    assert second
    assert second not in suffixes


class _ScreenOutput(DummyOutput):
    def __init__(self):
        self.chunks = []
        self.snapshots = []
        self._calls = 0

    def get_size(self):
        self._calls += 1
        import prompt_toolkit.data_structures

        return prompt_toolkit.data_structures.Size(rows=40, columns=80 + self._calls)

    def write(self, data):
        self.chunks.append(data)

    def flush(self):
        if self.chunks:
            self.snapshots.append("".join(self.chunks))
            self.chunks = []


def test_the_shadow_shows_the_fresh_roll_right_after_an_accept():
    mirror = GhostSuggestions(GhostChat(_session()))
    first = mirror.real_for("")
    mirror._chat.accept_rest("", first)
    mirror.forget()
    second = mirror.real_for(mirror._chat.session.text)
    assert second

    output = _ScreenOutput()
    with create_pipe_input() as pipe:
        failures = []

        def chat():
            try:
                run_chat(_session(), input=pipe, output=output)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=chat)
        thread.start()
        time.sleep(0.3)
        pipe.send_text("\t")
        time.sleep(0.5)
        pipe.send_text("\x03")
        thread.join(timeout=10)
    if failures:
        raise failures[0]

    rows = [
        snapshot.split("\r\n") for snapshot in output.snapshots if len(snapshot) > 1000
    ]
    assert rows, "the screen never rendered a full frame"
    assert first + second in "\n".join(row.rstrip("\r") for row in rows[-1])
