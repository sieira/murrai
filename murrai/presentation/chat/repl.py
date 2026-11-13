import contextlib

from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    FloatContainer,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from murrai.presentation.chat.form import FormOverlay, TrainingForm
from murrai.presentation.chat.lanes import (
    SILENT,
    CommandPicked,
    SessionSwapped,
)
from murrai.presentation.chat.panel import (
    CommandPanel,
    description_fragments,
    list_fragments,
)
from murrai.presentation.chat.status import (
    TrainingStatus,
    spawn_training,
    status_fragments,
)

GHOST_CHARS = 20
TOP = 5
STORY_TAIL_LINES = 20
PANEL_WIDTH = 36
CURSOR_GREEN = "\x1b]12;#00ff00\x07"
CURSOR_RESET = "\x1b]112\x07"
TOP_WEIGHT = 3
BOTTOM_WEIGHT = 1

STYLE = Style.from_dict(
    {
        "entered": "",
        "generated": "ansibrightcyan bold",
        "auto-suggestion": "ansibrightblack",
        "die-row": "ansibrightblack",
        "die-hit": "ansibrightcyan bold",
        "more": "ansibrightblack",
        "hint": "ansibrightblack",
        "selected": "ansibrightcyan bold",
        "bar": "ansigreen",
        "failed": "ansired bold",
        "dialog.body": "bg:default fg:default",
        "dialog.body text-area": "bg:default fg:default",
        "dialog.body text-area last-line": "nounderline",
    },
)


class GhostChat:
    def __init__(self, session, ghost_chars=GHOST_CHARS):
        self.session = session
        self.ghost_chars = ghost_chars
        self._synced = ""

    def use(self, session):
        self.session = session
        self._synced = session.text

    def sync(self, buffer_text: str) -> None:
        if self.session is None:
            return
        common = 0
        for a, b in zip(self._synced, buffer_text, strict=False):
            if a != b:
                break
            common += 1
        fresh = buffer_text[common:]
        self.session.grow(fresh)
        self.session.truncate(common)
        self.session.extend_user(fresh)
        self._synced = buffer_text

    def accept_rest(self, buffer_text: str, ghost: str) -> None:
        self.sync(buffer_text)
        self.session.extend_generated(ghost)
        self._synced = self.session.text

    def suggestions(self, buffer_text: str) -> str:
        if self.session is None:
            return ""
        try:
            return self.session.rollout(buffer_text, self.ghost_chars)
        except Exception:
            # a render must never kill the chat; prompt_toolkit wedges on any escape
            return ""

    def die_bar(self, buffer_text: str, ghost: str = ""):
        if self.session is None:
            return []
        try:
            dice = self.session.preview(buffer_text)
        except Exception:
            # a render must never kill the chat; prompt_toolkit wedges on any escape
            return []
        fragments = []
        hit = ghost[0] if ghost else None
        for char, p in dice[:TOP]:
            mark = "»" if char == hit else " "
            style = "class:die-hit" if char == hit else "class:die-row"
            shown = "↵" if char == "\n" else char
            fragments.append((style, f" {mark} {shown} {p:6.1%}"))
        if len(dice) > TOP:
            fragments.append(("class:more", f"   ... {len(dice) - TOP} more"))
        return fragments


class GhostSuggestions(AutoSuggest):
    def __init__(self, chat: GhostChat):
        self._chat = chat
        self._cache: dict[str, str] = {}
        self._chars: tuple = ()

    def _roll_for(self, context: str) -> str:
        chars = self._chat.session.chars if self._chat.session else ()
        if chars != self._chars:
            self._chars = chars
            self._cache.clear()
        if context not in self._cache:
            roll = self._chat.suggestions(context)
            if not roll:
                self._cache[context] = ""
                return ""
            for offset in range(len(roll) + 1):
                self._cache[context + roll[:offset]] = roll[offset:]
        return self._cache[context]

    def real_for(self, context: str) -> str:
        return self._roll_for(context)

    def get_suggestion(self, buffer, document):
        roll = self._roll_for(document.text)
        if not roll:
            return None
        return Suggestion(roll.replace("\n", "↵"))

    def forget(self) -> None:
        self._cache.clear()


def story_fragments(session, tail_lines=None):
    if session is None:
        return []
    segments = session.segments()
    if tail_lines is not None:
        kept = []
        lines_left = tail_lines
        for segment, origin in reversed(segments):
            pieces = segment.split("\n")
            take = pieces[::-1] if lines_left >= len(pieces) else pieces[-lines_left:]
            for piece in reversed(take):
                kept.append((origin, piece))
            lines_left -= len(pieces)
            if lines_left <= 0:
                break
        segments = [(text, origin) for origin, text in reversed(kept)]
    fragments = []
    for segment, origin in segments:
        style = "class:generated" if origin == "generated" else ""
        fragments.append((style, segment))
    return fragments


def _die_fragments(chat, suggestions, buffer):
    return chat.die_bar(buffer.text, suggestions.real_for(buffer.text))


def status_line(status, model_kind=None):
    runs = status_fragments(status)
    fragments = []
    if model_kind is not None:
        fragments.append(("class:selected", model_kind))
    if runs:
        if fragments:
            fragments.append(("class:hint", " │ "))
        fragments.extend(runs)
    return fragments or [("class:hint", "ctrl-p commands")]


def chat_layout(story, editor, die, panel_list, panel, overlay, status, chat):
    panel_column = HSplit(
        [
            Frame(panel_list, title="commands", height=Dimension(weight=TOP_WEIGHT)),
            Frame(
                Window(
                    FormattedTextControl(
                        lambda: description_fragments(panel, PANEL_WIDTH - 2),
                    ),
                ),
                title="about",
                height=Dimension(weight=BOTTOM_WEIGHT),
            ),
        ],
        width=PANEL_WIDTH,
    )
    return Layout(
        FloatContainer(
            HSplit(
                [
                    VSplit(
                        [
                            HSplit(
                                [
                                    Frame(
                                        story,
                                        title="murrAI",
                                        height=Dimension(weight=TOP_WEIGHT),
                                    ),
                                    Frame(
                                        HSplit([editor, die]),
                                        title="write",
                                        height=Dimension(weight=BOTTOM_WEIGHT),
                                    ),
                                ],
                            ),
                            ConditionalContainer(
                                panel_column,
                                Condition(lambda: panel.visible),
                            ),
                        ],
                    ),
                    Window(
                        FormattedTextControl(
                            lambda: status_line(
                                status,
                                chat.session.kind if chat.session else None,
                            ),
                        ),
                        height=1,
                        dont_extend_height=True,
                    ),
                ],
            ),
            floats=[overlay.float],
        ),
    )


def run_chat(
    session=None,
    commands=(),
    input=None,
    output=None,
    spawn=spawn_training,
    observer=SILENT,
    load=None,
):
    chat = GhostChat(session)
    suggestions = GhostSuggestions(chat)
    status = TrainingStatus(observer=observer)
    panel = CommandPanel(commands, observer)
    remembered: dict[str, dict] = {}

    editor = TextArea(
        text=session.text if session else "",
        auto_suggest=suggestions,
        multiline=True,
        complete_while_typing=False,
    )
    buffer = editor.buffer
    if session:
        buffer.cursor_position = len(session.text)

    def swap(next_session):
        observer.on_event(SessionSwapped())
        chat.use(next_session)
        suggestions.forget()
        buffer.text = next_session.text
        buffer.cursor_position = len(next_session.text)

    def schedule_swap(name):
        with contextlib.suppress(RuntimeError):
            # the app can quit before a build lands
            app.loop.call_soon_threadsafe(lambda: swap(load(name)))

    def focus(control):
        app.layout.focus(control)

    def accept_form(form, values):
        remembered[form.command.name] = values
        spawn(form.command.bake(values), status, schedule_swap)
        app.layout.focus(buffer)

    overlay = FormOverlay(
        on_accept=accept_form,
        on_cancel=lambda: focus(buffer),
        focus=focus,
    )

    def open_form(command):
        overlay.open(
            TrainingForm(
                command,
                remembered=remembered.get(command.name),
                observer=observer,
            ),
        )

    editor_keys = KeyBindings()

    @editor_keys.add("tab")
    def _(event):
        if chat.session is None:
            return
        real = suggestions.real_for(buffer.text)
        if not real:
            return
        try:
            chat.accept_rest(buffer.text, real)
        except ValueError:
            return
        suggestions.forget()
        buffer.text = chat.session.text
        buffer.cursor_position = len(buffer.text)
        buffer.suggestion = suggestions.get_suggestion(buffer, buffer.document)

    @editor_keys.add("enter")
    def _(event):
        if chat.session is not None:
            chat.sync(buffer.text)

    @editor_keys.add("escape", "[", "1", "3", ";", "2", "u")
    @editor_keys.add("escape", "c-m")
    def _(event):
        buffer.insert_text("\n")

    panel_keys = KeyBindings()

    @panel_keys.add("down")
    def _(event):
        panel.move(1)

    @panel_keys.add("up")
    def _(event):
        panel.move(-1)

    @panel_keys.add("enter")
    def _(event):
        command = panel.current
        if command is None:
            return
        panel.close()
        observer.on_event(CommandPicked(command.name))
        event.app.layout.focus(buffer)
        if command.trains:
            if load is None:
                status.fail(command.name, "no load function")
            elif command.fields:
                open_form(command)
            else:
                spawn(command, status, schedule_swap)
        else:
            swap(command.build())

    global_keys = KeyBindings()

    @global_keys.add("c-p")
    def _(event):
        if panel.visible:
            panel.close()
            event.app.layout.focus(buffer)
        elif overlay.visible:
            return
        else:
            panel.open()
            event.app.layout.focus(panel_list)

    @global_keys.add("escape", filter=Condition(lambda: panel.visible))
    def _(event):
        panel.close()
        event.app.layout.focus(buffer)

    @global_keys.add("c-c")
    def _(event):
        event.app.exit()

    story = Window(
        FormattedTextControl(
            lambda: (
                story_fragments(chat.session, tail_lines=STORY_TAIL_LINES)
                if chat.session
                else [("class:hint", "no model yet; ctrl-p opens commands")]
            ),
        ),
        wrap_lines=True,
    )
    die = Window(
        FormattedTextControl(lambda: _die_fragments(chat, suggestions, buffer)),
        height=1,
    )
    panel_list = Window(
        FormattedTextControl(
            lambda: list_fragments(panel),
            focusable=True,
            key_bindings=panel_keys,
        ),
    )
    layout = chat_layout(
        story=story,
        editor=editor,
        die=die,
        panel_list=panel_list,
        panel=panel,
        overlay=overlay,
        status=status,
        chat=chat,
    )
    editor.control.key_bindings = editor_keys
    app = Application(
        layout=layout,
        key_bindings=global_keys,
        style=STYLE,
        full_screen=True,
        cursor=CursorShape.BLOCK,
        input=input,
        output=output,
    )
    status.redraw = app.invalidate
    if session is None:
        panel.open()
        app.layout.focus(panel_list)
    app.output.write_raw(CURSOR_GREEN)
    app.output.flush()
    try:
        with contextlib.suppress(EOFError):
            app.run()
    finally:
        app.output.write_raw(CURSOR_RESET)
        app.output.flush()
    return chat.session


def plain_text(fragments):
    return "".join(text for _, text in fragments)
