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
