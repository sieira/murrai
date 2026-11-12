from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.ports import TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession, Command, run_chat
from murrai.presentation.chat.repl import GhostChat, GhostSuggestions


def _session(text="abab"):
    vocabulary = Vocabulary.from_text("ab")
    return ChatSession(Bigram.fit(text, vocabulary), 0)


def test_ctrl_p_opens_the_command_menu_and_swaps_the_session():
    built = []
    second = _session("baba")
    commands = [
        Command(
            "counted bigram",
            "odds tallied straight from the corpus",
            lambda: (built.append(1), second)[1],
        ),
    ]

    with create_pipe_input() as pipe:
        pipe.send_text("\x10counted bigram\r\x03")
        returned = run_chat(
            _session(),
            commands=commands,
            input=pipe,
            output=DummyOutput(),
        )

    assert built == [1]
    assert returned is second


def test_ctrl_p_without_a_command_keeps_the_session():
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x03")
        returned = run_chat(session, commands=[], input=pipe, output=DummyOutput())

    assert returned is session


def test_the_command_menu_rebuilds_each_time_it_opens():
    state = {"trained": False}
    trained = _session("baba")

    def commands():
        menu = []
        if state["trained"]:
            menu.append(
                Command(
                    "load counted-bigram",
                    "pick up this trained model",
                    lambda: trained,
                ),
            )
        menu.append(
            Command(
                "retrain counted bigram",
                "odds tallied straight from the corpus",
                lambda: (state.update(trained=True), trained)[1],
            ),
        )
        return menu

    with create_pipe_input() as pipe:
        pipe.send_text("\x10retrain counted bigram\r\x10load counted-bigram\r\x03")
        returned = run_chat(
            _session(),
            commands=commands,
            input=pipe,
            output=DummyOutput(),
        )

    assert state["trained"]
    assert returned is trained


def test_run_chat_without_a_session_opens_ready_to_pick():
    built = []
    model = _session("baba")
    commands = [
        Command(
            "counted bigram",
            "odds tallied straight from the corpus",
            lambda: (built.append(1), model)[1],
        ),
    ]

    with create_pipe_input() as pipe:
        pipe.send_text("counted bigram\r\x03")
        returned = run_chat(None, commands=commands, input=pipe, output=DummyOutput())

    assert built == [1]
    assert returned is model


def test_a_picked_training_command_runs_in_the_background():
    calls = []

    def build(on_event=None):
        calls.append(on_event)
        on_event(TrainingEvent(step=1, total_steps=10))
        return _session("baba")

    command = Command("retrain x", "slow model", build, trains=True)

    def spawn(cmd, status):
        cmd.build(on_event=status.track(cmd.name))

    first = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("\x10retrain x\r\x03")
        returned = run_chat(
            first,
            commands=[command],
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
        )

    assert len(calls) == 1
    assert calls[0] is not None
    assert returned is first


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


def test_tab_takes_the_rest_of_the_shadow():
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("a\t\x03")
        run_chat(session, input=pipe, output=DummyOutput())

    assert session.text.startswith("a")
    assert len(session.text) >= 10


def test_enter_commits_the_typed_text_without_taking_the_shadow():
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("ab\r\x03")
        run_chat(session, input=pipe, output=DummyOutput())

    assert session.text == "ab"


def test_the_story_pane_colors_generated_text():
    from murrai.presentation.chat.repl import story_fragments

    session = _session()
    session.extend_user("a")
    session.extend_generated("b")

    fragments = story_fragments(session)

    assert [style for style, _ in fragments] == ["", "class:generated"]
    assert "".join(text for _, text in fragments) == "ab"
