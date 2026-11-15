from prompt_toolkit.buffer import Buffer

from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import GhostChat, GhostSuggestions


def _session():
    return ChatSession(Bigram.fit("abab", TokenizerEncoderDecoder()), 0)


def test_a_second_tab_accept_rolls_fresh_ghost_text():
    chat = GhostChat(_session())
    suggestions = GhostSuggestions(chat)

    before = chat.session.text
    for _ in range(2):
        ghost = suggestions.real_for(chat.session.text)
        chat.accept_rest(chat.session.text, ghost)
        suggestions.forget()

    segments = chat.session.segments()
    assert [origin for _, origin in segments] == ["generated"]
    assert len(chat.session.text) > len(before)


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


def test_the_ghost_shows_the_fresh_roll_right_after_an_accept():
    chat = GhostChat(_session())
    suggestions = GhostSuggestions(chat)
    buffer = Buffer()

    first = suggestions.real_for(buffer.text)
    chat.accept_rest(buffer.text, first)
    suggestions.forget()
    buffer.text = chat.session.text
    buffer.cursor_position = len(buffer.text)
    buffer.suggestion = suggestions.get_suggestion(buffer, buffer.document)

    second = suggestions.real_for(buffer.text)
    assert buffer.suggestion is not None
    assert buffer.suggestion.text.replace("↵", "\n") == second
    suffixes = {first[k:] for k in range(len(first) + 1)}
    assert second not in suffixes
