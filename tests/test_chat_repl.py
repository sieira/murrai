import threading
import time

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from murrai.application.ports import TrainingEvent
from murrai.domain.bigram import Bigram
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat import ChatSession, Command, repl, run_chat
from murrai.presentation.chat.commands import FormCommand
from murrai.presentation.chat.form import START_FROM, Field
from murrai.presentation.chat.lanes import ChatLog
from murrai.presentation.chat.menu import AboutPopover
from murrai.presentation.chat.repl import GhostChat, GhostSuggestions


def _session(text="abab"):
    return ChatSession(Bigram.fit(text, TokenizerEncoderDecoder()), 0)


def test_ctrl_p_focuses_the_menu_and_swaps_the_session():
    built = []
    second = _session("baba")
    command = Command(
        "counted bigram",
        "odds tallied straight from the corpus",
        lambda: (built.append(1), second)[1],
    )

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r\x03")
        returned = run_chat(
            _session(),
            menu=lambda: {"new": (command,), "load": (), "retrain": (), "sessions": ()},
            input=pipe,
            output=DummyOutput(),
        )

    assert built == [1]
    assert returned is second


def test_arrows_choose_which_command_runs():
    built = []
    commands = (
        Command("first", "one", lambda: (built.append("first"), _session())[1]),
        Command("second", "two", lambda: (built.append("second"), _session())[1]),
    )

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\x1b[B\r\x03")
        run_chat(
            _session(),
            menu=lambda: {"new": commands, "load": (), "retrain": (), "sessions": ()},
            input=pipe,
            output=DummyOutput(),
        )

    assert built == ["second"]


def test_ctrl_p_without_a_command_keeps_the_session():
    session = _session()

    def empty():
        return {"new": (), "load": (), "retrain": (), "sessions": ()}

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x03")
        returned = run_chat(session, menu=empty, input=pipe, output=DummyOutput())

    assert returned is session


def test_the_command_menu_rebuilds_each_time_it_opens():
    state = {"trained": False}
    trained = _session("baba")

    def tree():
        new = (
            Command(
                "retrain counted bigram",
                "odds tallied straight from the corpus",
                lambda: (state.update(trained=True), trained)[1],
            ),
        )
        loads = ()
        if state["trained"]:
            loads = (
                Command(
                    "load counted-bigram",
                    "pick up this trained model",
                    lambda: trained,
                ),
            )
        return {"new": new, "load": loads, "retrain": (), "sessions": ()}

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.3)
        pipe.send_text("\x10\x1b[B\x1b[B\x1b[C\r")
        time.sleep(0.3)
        pipe.send_text("\x03")
        returned = run_chat(
            _session(),
            menu=tree,
            input=pipe,
            output=DummyOutput(),
        )

    assert state["trained"]
    assert returned is trained


def test_a_chat_without_a_model_opens_with_the_menu_active():
    built = []
    model = _session("baba")

    def tree():
        return {
            "new": (
                Command(
                    "counted bigram",
                    "odds tallied straight from the corpus",
                    lambda: (built.append(1), model)[1],
                ),
            ),
            "load": (),
            "retrain": (),
            "sessions": (),
        }

    output = _ScreenOutput()

    def feeder(pipe):
        time.sleep(0.5)
        for key in ("\x1b[B", "\x1b[C", "\r"):
            pipe.send_text(key)
            time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        returned = run_chat(None, menu=tree, input=pipe, output=output)
        thread.join()

    assert built == [1]
    assert returned is model


def test_the_menu_loads_a_checkpoint():
    loaded = []
    output = _ScreenOutput()

    def feeder(pipe):
        time.sleep(0.5)
        for key in ("\x1b[20~", "\x1b[B", "\x1b[B", "\x1b[C", "\r"):
            pipe.send_text(key)
            time.sleep(0.3)
        time.sleep(0.4)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(
            _session(),
            menu=lambda: {
                "new": (Command("counted bigram", "knobs", lambda: _session()),),
                "load": (
                    Command(
                        "load other",
                        "saved",
                        lambda: (loaded.append("other"), _session())[1],
                    ),
                ),
                "retrain": (),
                "sessions": (),
            },
            input=pipe,
            output=output,
            load=lambda name: (loaded.append(name), _session())[1],
        )
        thread.join()

    assert loaded == ["other"]


def test_picking_retrain_from_the_menu_pins_the_items_own_checkpoint(monkeypatch):
    from murrai.presentation.chat import repl
    from murrai.presentation.chat.form import START_FROM, TrainingForm

    opened = []
    real_init = TrainingForm.__init__

    class Spy(TrainingForm):
        def __init__(self, command, remembered=None, observer=None, pinned=None):
            real_init(
                self,
                command,
                remembered=remembered,
                observer=observer,
                pinned=pinned,
            )
            opened.append((command.name, pinned))

    monkeypatch.setattr(repl, "TrainingForm", Spy)
    session = ChatSession(
        Bigram.fit("abab", TokenizerEncoderDecoder()),
        0,
        checkpoint="char-gpt",
    )
    output = _ScreenOutput()

    def feeder(pipe):
        time.sleep(0.5)
        for key in (
            "\x1b[20~",
            "\x1b[B",
            "\x1b[B",
            "\x1b[B",
            "\x1b[C",
            "\r",
            "\x1b",
        ):
            pipe.send_text(key)
            time.sleep(0.3)
        time.sleep(0.4)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(
            session,
            menu=lambda: {
                "new": (Command("counted bigram", "knobs", lambda: _session()),),
                "load": (),
                "retrain": (
                    _menu_form_command(
                        "retrain gpt-corpus",
                        pin={START_FROM: "gpt-corpus"},
                    ),
                ),
                "sessions": (),
            },
            input=pipe,
            output=output,
            load=lambda name: _session(),
        )
        thread.join()

    assert opened[0][0] == "retrain gpt-corpus"
    assert opened[0][1] == {START_FROM: "gpt-corpus"}


def test_a_training_command_without_a_load_function_fails_loudly():
    spawned = []
    log = ChatLog()
    command = Command("retrain x", "slow model", lambda: _session(), trains=True)

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r\x03")
        run_chat(
            _session(),
            menu=lambda: {"new": (command,), "load": (), "retrain": (), "sessions": ()},
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
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.3)
        pipe.send_text("\x03")
        returned = run_chat(
            first,
            menu=lambda: {"new": (command,), "load": (), "retrain": (), "sessions": ()},
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
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.7)
        pipe.send_text("\x10\x1b[B\x1b[B\x1b[C\r")
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
            menu=lambda: {
                "new": (command,),
                "load": (
                    Command(
                        "load prepared",
                        "pick up this trained model",
                        lambda: _session(prepared.text),
                    ),
                ),
                "retrain": (),
                "sessions": (),
            },
            input=pipe,
            output=DummyOutput(),
            observer=log,
            load=lambda name: _session(prepared.text),
        )
        thread.join()

    lines = [line.split(" ", 1)[1] for line in log.lines]
    picked = lines.index("[command] picked retrain x")
    mid_pick = lines.index("[command] picked load prepared")
    swapped = next(i for i, line in enumerate(lines) if line == "[session] swapped")
    assert picked < mid_pick < swapped
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


def test_tab_takes_the_rest_of_the_ghost():
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("a\t\x03")
        run_chat(session, input=pipe, output=DummyOutput())

    assert session.text.startswith("a")
    assert len(session.text) >= 10


def test_enter_commits_the_typed_text_without_taking_the_ghost():
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
        run_chat(_session(), input=pipe, output=output)

    assert output.raw[0] == "\x1b]12;#00ff00\x07"
    assert output.raw[-1] == "\x1b]112\x07"


def _form_session(text="abab5x"):
    return ChatSession(Bigram.fit(text, TokenizerEncoderDecoder()), 0)


def _steps_field(**extra):
    return Field("steps", "steps", "int", 2000, check="whole above zero", **extra)


ACCEPTED = {}


def _menu_form_command(name, continues=True, pin=None):
    def bake(values):
        ACCEPTED.clear()
        ACCEPTED.update(values)
        return Command(name, "knobs", lambda: _form_session(), trains=True)

    return FormCommand(
        name,
        "knobs",
        lambda: None,
        fields=(_steps_field(),),
        bake=bake,
        trains=True,
        continues=continues,
        pin=pin,
    )


def _form_command(fields=None, continues=False, pin=None):
    def bake(values):
        ACCEPTED.clear()
        ACCEPTED.update(values)
        return Command(
            "retrain x",
            "knobs",
            lambda: _form_session(),
            trains=True,
        )

    return FormCommand(
        "retrain x",
        "knobs",
        lambda: None,
        fields=fields or (_steps_field(),),
        bake=bake,
        trains=True,
        continues=continues,
        pin=pin,
    )


def _form_chat(menu, spawn, log=None, load=None, checkpoint=None):
    def run(pipe):
        return run_chat(
            ChatSession(
                Bigram.fit("abab5x", TokenizerEncoderDecoder()),
                0,
                checkpoint=checkpoint,
            ),
            menu=menu,
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            observer=log or ChatLog(),
            load=load or (lambda name: _form_session()),
        )

    return run


def _save_command(values):
    ACCEPTED.clear()
    ACCEPTED.update(values)
    return Command(
        "save story",
        "knobs",
        lambda: _form_session(),
        trains=False,
    )


def test_picking_save_story_opens_the_form_and_accepting_runs_it_synchronously():
    spawned = []
    log = ChatLog(clock=lambda: 0.0)
    command = FormCommand(
        "save story",
        "write this story down to load it later",
        lambda: _form_session(),
        fields=(
            Field(
                "name",
                "name",
                "text",
                "gpt-corpus-story",
                check="at least one character",
            ),
        ),
        bake=lambda values: _save_command(values),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[C\x1b[B\r")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {"new": (), "load": (), "retrain": (), "sessions": (command,)},
            spawn,
            log,
        )(pipe)

    assert spawned == []
    assert ACCEPTED == {"name": "gpt-corpus-story"}
    assert "[form] opened save story" in [line.split(" ", 1)[1] for line in log.lines]


def test_a_swap_tells_the_swap_listener():
    swapped = []

    def feeder(pipe):
        time.sleep(0.5)
        for key in ("\x1b[20~", "\x1b[B", "\x1b[B", "\x1b[C", "\r"):
            pipe.send_text(key)
            time.sleep(0.3)
        time.sleep(0.4)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(
            _session(),
            menu=lambda: {
                "new": (),
                "load": (Command("load other", "saved", lambda: _session()),),
                "retrain": (),
                "sessions": (),
            },
            input=pipe,
            output=_ScreenOutput(),
            load=lambda name: _session(),
            on_swap=swapped.append,
        )
        thread.join()

    assert len(swapped) == 1
    assert isinstance(swapped[0], ChatSession)


def test_picking_a_retrain_command_opens_a_form_and_escape_returns_to_the_editor():
    spawned = []
    log = ChatLog(clock=lambda: 0.0)
    command = _form_command()

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\x10")
        time.sleep(0.2)
        pipe.send_text("\x1b")
        time.sleep(0.2)
        pipe.send_text("\x10")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {"new": (command,), "load": (), "retrain": (), "sessions": ()},
            spawn,
            log,
        )(
            pipe,
        )

    assert spawned == []
    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[command] picked retrain x",
        "[form] opened retrain x",
        "[form] cancelled retrain x",
    ]


def test_the_form_spawns_the_command_with_the_edited_values():
    spawned = []

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\x7f\x7f\x7f\x7f5")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {
                "new": (_form_command(),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            spawn,
        )(pipe)

    assert len(spawned) == 1
    assert ACCEPTED == {"steps": 5}
    assert spawned[0].name == "retrain x"
    assert spawned[0].trains


def test_the_form_remembers_edited_values_for_the_session():
    spawned = []

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\x7f\x7f\x7f\x7f5\r")
        time.sleep(0.2)
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {
                "new": (_form_command(),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            spawn,
        )(pipe)

    assert ACCEPTED == {"steps": 5}
    assert len(spawned) == 2


def test_a_bad_value_keeps_the_form_open_and_spawns_nothing():
    spawned = []
    log = ChatLog(clock=lambda: 0.0)

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("x")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x1b")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {
                "new": (_form_command(),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            spawn,
            log,
        )(pipe)

    assert spawned == []
    assert [line.split(" ", 1)[1] for line in log.lines][-2:] == [
        "[form] opened retrain x",
        "[form] cancelled retrain x",
    ]


def test_the_retrain_form_leaves_the_shape_to_the_savegame():
    fields = (
        _steps_field(),
        Field("depth", "depth", "int", 4, check="whole above zero", shape=True),
    )

    def spawn(cmd, status, on_built):
        pass

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[B\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {
                "new": (),
                "load": (),
                "retrain": (
                    _form_command(
                        fields,
                        continues=True,
                        pin={START_FROM: "char-gpt"},
                    ),
                ),
                "sessions": (),
            },
            spawn,
        )(pipe)

    assert ACCEPTED == {"steps": 2000, "start_from": "char-gpt"}


def test_a_form_command_without_a_load_function_fails_at_the_pick():
    log = ChatLog(clock=lambda: 0.0)

    def spawn(cmd, status, on_built):
        pass

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        run_chat(
            _form_session(),
            menu=lambda: {
                "new": (_form_command(),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            input=pipe,
            output=DummyOutput(),
            spawn=spawn,
            observer=log,
        )

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[command] picked retrain x",
        "[train] retrain x failed: no load function",
    ]


def test_a_story_load_whose_model_is_gone_fails_at_the_pick():
    log = ChatLog(clock=lambda: 0.0)

    def broken():
        raise KeyError("gpt-corpus")

    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        for key in ("\x1b[B", "\x1b[B", "\x1b[C", "\r"):
            pipe.send_text(key)
            time.sleep(0.2)
        time.sleep(0.2)
        pipe.send_text("\x03")
        run_chat(
            _form_session(),
            menu=lambda: {
                "new": (),
                "load": (Command("load hello-story", "saved", broken),),
                "retrain": (),
                "sessions": (),
            },
            input=pipe,
            output=DummyOutput(),
            observer=log,
        )

    assert [line.split(" ", 1)[1] for line in log.lines] == [
        "[command] picked load hello-story",
        "[train] load hello-story failed: the model gpt-corpus is gone",
    ]


def test_the_right_arrow_moves_to_the_next_choice():
    spawned = []
    fields = (
        _steps_field(),
        Field("size", "size", "choice", "all", choices=("all", "100k")),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
        time.sleep(0.2)
        pipe.send_text("\t")
        time.sleep(0.2)
        pipe.send_text("\x1b[C")
        time.sleep(0.2)
        pipe.send_text("\r")
        time.sleep(0.2)
        pipe.send_text("\x03")
        _form_chat(
            lambda: {
                "new": (_form_command(fields),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            spawn,
        )(pipe)

    assert ACCEPTED == {"steps": 2000, "size": "100k"}


def test_the_left_arrow_moves_to_the_previous_choice():
    spawned = []
    fields = (
        _steps_field(),
        Field("size", "size", "choice", "all", choices=("all", "100k")),
    )

    def spawn(cmd, status, on_built):
        spawned.append(cmd)

    with create_pipe_input() as pipe:
        pipe.send_text("\x10\x1b[B\x1b[C\r")
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
        _form_chat(
            lambda: {
                "new": (_form_command(fields),),
                "load": (),
                "retrain": (),
                "sessions": (),
            },
            spawn,
        )(pipe)

    assert ACCEPTED == {"steps": 2000, "size": "all"}


def _chat_with_tabs(count):
    session = _session()
    with create_pipe_input() as pipe:
        pipe.send_text("a" + "\t" * count + "\x03")
        return run_chat(session, input=pipe, output=DummyOutput())


def test_a_second_tab_takes_a_fresh_ghost():
    one = _chat_with_tabs(1)
    two = _chat_with_tabs(2)

    assert len(two.text) > len(one.text)


def _broken_line_session(text):
    return ChatSession(Bigram.fit(text, TokenizerEncoderDecoder()), 0)


def test_the_story_renders_typed_line_breaks_as_lines():
    from murrai.presentation.chat.repl import story_fragments

    session = _broken_line_session("ab\ncd")
    session.extend_user("ab\ncd")

    fragments = story_fragments(session)

    assert "".join(text for _, text in fragments) == "ab\ncd"


def test_the_story_tail_keeps_the_last_lines_intact():
    from murrai.presentation.chat.repl import story_fragments

    session = _broken_line_session("ab\ncd\nef")
    session.extend_user("ab\ncd\nef")

    fragments = story_fragments(session, tail_lines=2)

    assert "".join(text for _, text in fragments) == "cd\nef"


class _ScreenOutput(DummyOutput):
    def __init__(self):
        self.chunks = []
        self.snapshots = []
        self._calls = 0

    def get_size(self):
        self._calls += 1
        from prompt_toolkit.data_structures import Size

        return Size(rows=40, columns=80 + self._calls)

    def write(self, data):
        self.chunks.append(data)

    def flush(self):
        if self.chunks:
            self.snapshots.append("".join(self.chunks))
            self.chunks = []


def _wait_for_row(output, text, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        large = [s for s in output.snapshots if len(s) > 1000]
        if large:
            shown = [row.strip("│ \r") for row in large[-1].split("\r\n")]
            if text in shown:
                return large[-1]
        time.sleep(0.05)
    raise AssertionError(f"no row {text!r} showed up on the screen")


def _run_with_popover_spy(monkeypatch, keys):
    popovers = []

    class Spy(AboutPopover):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            popovers.append(self)

    monkeypatch.setattr(repl, "AboutPopover", Spy)
    session = _session()
    output = _ScreenOutput()

    def feeder(pipe):
        time.sleep(0.5)
        for chunk in keys:
            pipe.send_text(chunk)
            time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(session, input=pipe, output=output)
        thread.join()
    return popovers[0]


F9 = "\x1b[20~"
ABOUT_ITEM = "\x1b[C\x1b[C\x1b[B"


def test_f9_leaves_an_open_about_popover_alone(monkeypatch):
    popover = _run_with_popover_spy(
        monkeypatch,
        [F9, ABOUT_ITEM, "\r", F9, "\x1b"],
    )

    assert popover.visible is False


def test_space_picks_the_highlighted_menu_item(monkeypatch):
    popover = _run_with_popover_spy(monkeypatch, [F9, ABOUT_ITEM, " "])

    assert popover.visible is True


def test_f9_opens_the_menu_bar_arrows_walk_and_about_picks():
    session = _session()
    output = _ScreenOutput()

    def feeder(pipe):
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\x1b[C\x1b[C\x1b[B")
        time.sleep(0.3)
        pipe.send_text("\r")
        time.sleep(0.3)
        pipe.send_text("\x1b")
        time.sleep(0.3)
        pipe.send_text("xy\r")
        time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(session, input=pipe, output=output)
        thread.join()

    assert any("built from scratch" in snapshot for snapshot in output.snapshots)
    assert session.text == "xy"


def test_ctrl_c_inside_the_menu_bar_quits_the_chat():
    session = _session()
    output = _ScreenOutput()
    done = []

    with create_pipe_input() as pipe:

        def chat():
            run_chat(session, input=pipe, output=output)
            done.append(1)

        thread = threading.Thread(target=chat)
        thread.start()
        time.sleep(0.5)
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\x03")
        thread.join(timeout=10)
        if thread.is_alive():
            pipe.send_text("\x03")
            thread.join(timeout=5)
            raise AssertionError("ctrl-c inside the menu bar did not quit")
    assert done == [1]


def test_a_click_outside_the_bar_deselects_it():
    session = _session()
    output = _ScreenOutput()

    def feeder(pipe):
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\x1b[<0;10;4M")
        time.sleep(0.1)
        pipe.send_text("\x1b[<0;10;4m")
        time.sleep(0.3)
        pipe.send_text("xy\r")
        time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(session, input=pipe, output=output)
        thread.join()

    assert session.text == "xy"
    assert not any("built from scratch" in snapshot for snapshot in output.snapshots)


def test_f9_keeps_typing_out_of_the_editor_while_the_bar_is_open():
    session = _session()

    def feeder(pipe):
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\t")
        time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(session, input=pipe, output=DummyOutput())
        thread.join()

    assert session.text == ""


def test_a_second_f9_closes_the_bar_and_returns_to_the_editor():
    session = _session()

    def feeder(pipe):
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\x1b[20~")
        time.sleep(0.3)
        pipe.send_text("\t")
        time.sleep(0.3)
        pipe.send_text("\x03")

    with create_pipe_input() as pipe:
        thread = threading.Thread(target=feeder, args=(pipe,))
        thread.start()
        run_chat(session, input=pipe, output=DummyOutput())
        thread.join()

    assert len(session.text) > len("abab")


def test_the_story_fills_the_box_with_one_row_per_line():
    session = _broken_line_session("\n".join("abcdefghijkl"))
    session.extend_user("\n".join("abcdefghijkl"))
    output = _ScreenOutput()
    with create_pipe_input() as pipe:
        failures = []

        def chat():
            try:
                run_chat(session, input=pipe, output=output)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=chat)
        thread.start()
        full = _wait_for_row(output, "a")
        pipe.send_text("\x03")
        thread.join(timeout=10)
    if failures:
        raise failures[0]

    shown = [row.strip("│ \r") for row in full.split("\r\n")]
    for letter in "abcde":
        assert letter in shown


def test_the_story_box_uses_its_height_not_a_fixed_twenty_lines():
    first = "\n".join("abcd")
    session = _broken_line_session(
        "abcd"
        + "\n"
        + "e\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz",
    )
    session.extend_user(first)
    output = _ScreenOutput()
    with create_pipe_input() as pipe:
        failures = []

        def chat():
            try:
                run_chat(session, input=pipe, output=output)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=chat)
        thread.start()
        _wait_for_row(output, "d")
        sequence = "\x1b[13;2u"
        typed = "".join(sequence + letter for letter in "fghijklmnopqrstuvwxyz")
        pipe.send_text("e" + typed + "\r")
        full = _wait_for_row(output, "f")
        pipe.send_text("\x03")
        thread.join(timeout=10)
    if failures:
        raise failures[0]

    shown = [row.strip("│ \r") for row in full.split("\r\n")]
    visible = [c for c in "abcdefghijklmnopqrstuvwx" if c in shown]
    assert len(visible) >= 21
