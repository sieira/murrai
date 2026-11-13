import threading
import time

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.ports import TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession, Command, run_chat
from murrai.presentation.chat.form import Field
from murrai.presentation.chat.lanes import ChatLog
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
        pipe.send_text("\x10\r\x03")
        returned = run_chat(
            _session(),
            commands=commands,
            input=pipe,
            output=DummyOutput(),
        )

    assert built == [1]
    assert returned is second


def test_arrows_choose_which_command_runs():
    built = []
    commands = [
        Command("first", "one", lambda: (built.append("first"), _session())[1]),
        Command("second", "two", lambda: (built.append("second"), _session())[1]),
    ]

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\r\x03")
        run_chat(_session(), commands=commands, input=pipe, output=DummyOutput())

    assert built == ["second"]


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
        pipe.send_text("\x10\r\x10\r\x03")
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
        pipe.send_text("\r\x03")
        returned = run_chat(None, commands=commands, input=pipe, output=DummyOutput())

    assert built == [1]
    assert returned is model


def test_a_training_command_without_a_load_function_fails_loudly():
    spawned = []
    log = ChatLog()
    command = Command("retrain x", "slow model", lambda: _session(), trains=True)

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r\x03")
        run_chat(
            _session(),
            commands=[command],
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            observer=log,
        )

    assert spawned == []
    assert any("[train] retrain x failed" in line for line in log.lines)


def test_a_picked_training_command_swaps_in_the_loaded_checkpoint():
    calls = []
    names = []
    loaded = []
    session = _session("baba")

    def build(on_event=None):
        calls.append(on_event)
        on_event(TrainingEvent(step=1, total_steps=10))
        return session

    command = Command("retrain x", "slow model", build, trains=True)

    def spawn(cmd, status, on_built):
        on_built(cmd.build(on_event=status.track(cmd.name)).kind)

    def load(name):
        names.append(name)
        swap = _session("abba")
        loaded.append(swap)
        return swap

    first = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.3)
        pipe.send_text("\x03")
        returned = run_chat(
            first,
            commands=[command],
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            load=load,
        )

    assert len(calls) == 1
    assert calls[0] is not None
    assert names == ["counted-bigram"]
    assert returned is loaded[0]


class SlowBuilder:
    def __init__(self, seconds, session):
        self.seconds = seconds
        self.session = session

    def build(self, on_event=None):
        time.sleep(self.seconds)
        return self.session


def test_the_screen_answers_keys_while_a_training_builds():
    prepared = _session("baba")
    builder = SlowBuilder(1.5, prepared)
    command = Command(
        "retrain x",
        "slow model",
        builder.build,
        trains=True,
    )
    log = ChatLog()

    def feeder(pipe):
        pipe.send_text("\x10\r")
        time.sleep(0.7)
        pipe.send_text("\x10")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any("[session] swapped" in line for line in log.lines):
                break
            time.sleep(0.05)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        returned = run_chat(
            _session(),
            commands=[command],
            input=pipe,
            output=DummyOutput(),
            observer=log,
            load=lambda name: _session(prepared.text),
        )
        thread.join()

    lines = [line.split(" ", 1)[1] for line in log.lines]
    picked = lines.index("[command] picked retrain x")
    swapped = next(i for i, line in enumerate(lines) if line == "[session] swapped")
    assert lines[picked:swapped].count("[panel] opened 1 commands") == 1
    assert returned.text == prepared.text
    assert returned is not prepared


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


class _RecordingOutput(DummyOutput):
    def __init__(self):
        self.raw = []

    def write_raw(self, data):
        self.raw.append(data)


def test_the_status_bars_are_green():
    from murrai.presentation.chat.repl import STYLE

    assert STYLE.get_attrs_for_style_str("class:bar").color == "ansigreen"


def test_the_cursor_turns_green_while_chatting_and_resets_on_exit():
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        pipe.send_text("\x03")
        run_chat(_session(), commands=[], input=pipe, output=output)

    assert output.raw[0] == "\x1b]12;#00ff00\x07"
    assert output.raw[-1] == "\x1b]112\x07"


def _form_session(text="abab5x"):
    return ChatSession(Bigram.fit(text, Vocabulary.from_text(text)), 0)


def _steps_field(**extra):
    return Field("steps", "steps", "int", 2000, check="whole above zero", **extra)


def _form_command(fields=None):
    def bake(values):
        return Command(
            "retrain x",
            "knobs",
            lambda: _form_session(),
            trains=True,
            given=values,
        )

    return Command(
        "retrain x",
        "knobs",
        lambda: None,
        trains=True,
        fields=fields or (_steps_field(),),
        bake=bake,
    )


def _form_chat(commands, spawn, log=None, load=None):
    def run(pipe):
        return run_chat(
            _form_session(),
            commands=commands,
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            observer=log or ChatLog(),
            load=load or (lambda name: _form_session()),
        )

    return run


def test_picking_a_retrain_command_opens_a_form_and_escape_returns_to_the_editor():
    spawned = []
    log = ChatLog(clock=lambda: 0.0)
    command = _form_command()

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\x10")
        time.sleep(0.2)
        pipe.send_text("\x1b")
        time.sleep(0.2)
        pipe.send_text("\x10")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([command], spawn, log)(pipe)

    assert spawned == []
    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[panel] opened 1 commands",
        "[panel] closed",
        "[command] picked retrain x",
        "[form] opened retrain x",
        "[form] cancelled retrain x",
        "[panel] opened 1 commands",
    ]


def test_the_form_spawns_the_command_with_the_edited_values():
    spawned = []

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\x7f\x7f\x7f\x7f5")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command()], spawn)(pipe)

    assert len(spawned) == 1
    assert spawned[0].given == {"steps": 5}
    assert spawned[0].name == "retrain x"
    assert spawned[0].trains


def test_the_form_remembers_edited_values_for_the_session():
    spawned = []

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\x7f\x7f\x7f\x7f5\r")
        time.sleep(0.2)
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command()], spawn)(pipe)

    assert [run.given for run in spawned] == [{"steps": 5}, {"steps": 5}]


def test_a_bad_value_keeps_the_form_open_and_spawns_nothing():
    spawned = []
    log = ChatLog(clock=lambda: 0.0)

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("x")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x1b")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command()], spawn, log)(pipe)

    assert spawned == []
    assert [line.split(" ", 1)[1] for line in log.lines][-2:] == [
        "[form] opened retrain x",
        "[form] cancelled retrain x",
    ]


def test_the_form_leaves_the_shape_out_of_a_savegame_run():
    spawned = []
    fields = (
        _steps_field(),
        Field(
            "start_from",
            "start from",
            "choice",
            "fresh",
            choices=("fresh", "char-gpt"),
        ),
        Field("depth", "depth", "int", 4, check="whole above zero", shape=True),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\x1b[B")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command(fields)], spawn)(pipe)

    assert spawned[0].given == {"steps": 2000, "start_from": "char-gpt"}


def test_a_form_command_without_a_load_function_fails_at_the_pick():
    log = ChatLog(clock=lambda: 0.0)

    def spawn(cmd, status, on_built):
        pass

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        run_chat(
            _form_session(),
            commands=[_form_command()],
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            observer=log,
        )

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[panel] opened 1 commands",
        "[panel] closed",
        "[command] picked retrain x",
        "[train] retrain x failed: no load function",
    ]


def test_the_right_arrow_moves_to_the_next_choice():
    spawned = []
    fields = (
        _steps_field(),
        Field(
            "start_from",
            "start from",
            "choice",
            "fresh",
            choices=("fresh", "char-gpt"),
        ),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\x1b[C")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command(fields)], spawn)(pipe)

    assert spawned[0].given == {"steps": 2000, "start_from": "char-gpt"}


def test_the_left_arrow_moves_to_the_previous_choice():
    spawned = []
    fields = (
        _steps_field(),
        Field(
            "start_from",
            "start from",
            "choice",
            "fresh",
            choices=("fresh", "char-gpt"),
        ),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\r")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\x1b[B")
        time.sleep(0.2)
        pipe.send_text("\x1b[D")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat([_form_command(fields)], spawn)(pipe)

    assert spawned[0].given == {"steps": 2000, "start_from": "fresh"}
