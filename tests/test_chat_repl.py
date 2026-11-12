from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession, Stage, run_chat
from murrai.presentation.chat.repl import GhostChat, GhostSuggestions


def _session(text="abab"):
    vocabulary = Vocabulary.from_text("ab")
    return ChatSession(Bigram.fit(text, vocabulary), 0)


def test_ctrl_p_opens_the_command_menu_and_swaps_the_session():
    built = []
    second = _session("baba")
    commands = [
        Stage(
            "retrain counted bigram",
            "odds tallied straight from the corpus",
            lambda: (built.append(1), second)[1],
        )
    ]

    with create_pipe_input() as pipe:
        pipe.send_text("\x10retrain counted bigram\r\x04")
        run_chat(_session(), commands=commands, input=pipe, output=DummyOutput())

    assert built == [1]


def test_ctrl_p_without_a_command_keeps_the_session():
    with create_pipe_input() as pipe:
        pipe.send_text("\x04")
        run_chat(_session(), commands=[], input=pipe, output=DummyOutput())


def test_ghost_chat_takes_a_new_session():
    chat = GhostChat(_session())
    following = _session("baba")

    chat.use(following)

    assert chat.session is following
    chat.sync("")
    assert chat.session.text == ""


def test_ghost_suggestions_forgets_the_old_model():
    chat = GhostChat(_session())
    suggestions = GhostSuggestions(chat)
    suggestions._roll_for("")
    assert suggestions._cache

    suggestions.forget()

    assert suggestions._cache == {}
