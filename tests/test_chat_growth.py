import contextlib
import io
import threading

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.runs import train_neural_bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import (
    GhostChat,
    GhostSuggestions,
    _die_fragments,
    run_chat,
)


def _neural_session(text="abab"):
    model = train_neural_bigram(text, steps=3, batch=4, seed=0).model
    return ChatSession(model, seed=0)


def _piped(session, keys, seconds=10):
    box = []

    def drive():
        sink = io.StringIO()
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            box.append(run_chat(session, input=pipe, output=DummyOutput()))

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        worker = threading.Thread(target=drive, daemon=True)
        worker.start()
        worker.join(seconds)
    return worker, box


def test_typing_an_unknown_char_grows_the_model_instead_of_killing_the_chat():
    session = _neural_session()

    worker, box = _piped(session, "abé\r\x03")

    assert not worker.is_alive(), "the chat wedged on the unknown character"
    assert box == [session]
    assert "é" in session.text
    assert "é" in session.chars


class _WideRow:
    kind = "wide-row"

    def __init__(self):
        self.vocabulary = Vocabulary.from_text("ab")

    def probs(self, context_ids):
        return [0.25, 0.25, 0.5]


def test_the_ghost_and_the_die_bar_survive_a_model_that_misbehaves():
    chat = GhostChat(ChatSession(_WideRow(), seed=0))
    suggestions = GhostSuggestions(chat)

    class _Buffer:
        text = "a"

    assert suggestions.get_suggestion(None, Document("a")) is None
    assert isinstance(_die_fragments(chat, suggestions, _Buffer()), list)


def test_a_growth_refreshes_the_cached_suggestions():
    chat = GhostChat(_neural_session())
    suggestions = GhostSuggestions(chat)
    assert suggestions.real_for("abé") == ""

    chat.sync("abé")

    assert suggestions.real_for("abé") != ""


def test_a_chat_growth_keeps_the_old_answers():
    chat = GhostChat(_neural_session())
    before = [char for char, _ in chat.session.preview("ab")]

    chat.sync("abé")

    after = [char for char, _ in chat.session.preview("ab")]
    assert [char for char in after if char in before] == before
