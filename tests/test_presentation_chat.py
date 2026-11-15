import pytest
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat import ChatSession
from murrai.presentation.chat.repl import (
    GhostChat,
    GhostSuggestions,
    plain_text,
    run_chat,
)


@pytest.fixture
def model():
    return Bigram.fit("abab", TokenizerEncoderDecoder())


@pytest.fixture
def chat(model):
    return GhostChat(ChatSession(model, seed=0))


def test_the_ghost_is_a_twenty_token_rollout(chat):
    suggestion = GhostSuggestions(chat).get_suggestion(None, Document("a"))
    assert isinstance(suggestion, Suggestion)
    assert len(suggestion.text) >= 20


def test_the_ghost_survives_text_absent_from_the_corpus(chat):
    suggestion = GhostSuggestions(chat).get_suggestion(None, Document("az"))
    assert suggestion is not None


def test_the_ghost_advances_after_an_accept_instead_of_rerolling(chat):
    suggestions = GhostSuggestions(chat)
    first = suggestions.get_suggestion(None, Document("a")).text
    after_one = suggestions.get_suggestion(None, Document("a" + first[0])).text
    assert after_one == first[1:]


def test_the_ghost_is_deterministic_per_seed(model):
    one = GhostChat(ChatSession(model, seed=3))
    two = GhostChat(ChatSession(model, seed=3))
    assert one.suggestions("ab") == two.suggestions("ab")


@pytest.fixture
def newline_chat():
    model = Bigram.fit("a\nb\na\nb", TokenizerEncoderDecoder())
    return GhostChat(ChatSession(model, seed=0))


def test_the_ghost_shows_newlines_as_a_marker(newline_chat):
    real = newline_chat.suggestions("")
    if "\n" not in real:
        pytest.skip("rollout held no newline")
    suggestions = GhostSuggestions(newline_chat)
    shown = suggestions.get_suggestion(None, Document("")).text
    assert "\n" not in shown
    assert "↵" in shown


def test_the_ghost_keeps_the_real_roll_for_acceptance(newline_chat):
    suggestions = GhostSuggestions(newline_chat)
    shown = suggestions.get_suggestion(None, Document("")).text
    real = suggestions.real_for("")
    assert shown.replace("↵", "\n") == real


def test_the_story_holds_newlines(newline_chat):
    newline_chat.sync("a\nb")
    assert newline_chat.session.text == "a\nb"
    assert newline_chat.session.segments() == [("a\nb", "entered")]


def test_sync_marks_typed_chars_as_entered(chat):
    chat.sync("ab")
    assert chat.session.text == "ab"
    assert chat.session.segments() == [("ab", "entered")]


def test_sync_survives_a_deletion(chat):
    chat.sync("ab")
    chat.sync("a")
    assert chat.session.text == "a"
    assert chat.session.segments() == [("a", "entered")]


def test_accept_rest_lands_the_whole_suggestion(chat):
    ghost = chat.suggestions("a")
    chat.sync("a")
    chat.accept_rest("a", ghost)
    assert chat.session.text == "a" + ghost
    assert chat.session.segments()[-1] == (ghost, "generated")


def test_the_die_bar_marks_what_tab_will_accept(chat):
    ids, ghost = chat.roll("a")
    chat.sync("a")
    bar = chat.die_bar("a", ids[0])
    text = plain_text(bar)
    assert chat.session.text
    assert ghost
    assert "»" in text
    assert "%" in text


def test_the_die_bar_survives_text_absent_from_the_corpus(chat):
    assert plain_text(chat.die_bar("az")) != ""


def test_run_chat_types_and_returns_the_story(chat):
    with create_pipe_input() as pipe:
        pipe.send_text("ab\r")
        pipe.close()
        run_chat(chat.session, input=pipe, output=DummyOutput())
    assert chat.session.text == "ab"
    assert chat.session.segments() == [("ab", "entered")]


def test_run_chat_ctrl_c_quits(chat):
    with create_pipe_input() as pipe:
        pipe.send_text("a\x03")
        pipe.close()
        run_chat(chat.session, input=pipe, output=DummyOutput())
    assert chat.session.text == ""
