import contextlib

from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

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
        self.session.truncate(common)
        self.session.extend_user(buffer_text[common:])
        self._synced = buffer_text

    def accept_rest(self, buffer_text: str, ghost: str) -> None:
        self.sync(buffer_text)
        self.session.extend_generated(ghost)
        self._synced = self.session.text

    def suggestions(self, buffer_text: str) -> str:
        if self.session is None:
            return ""
        return self.session.rollout(buffer_text, self.ghost_chars)

    def roll(self, buffer_text: str) -> tuple[list[int], str]:
        if self.session is None:
            return [], ""
        return self.session.rollout_ids(buffer_text, self.ghost_chars)

    def die_bar(self, buffer_text: str, next_id: int | None = None):
        if self.session is None:
            return []
        dice = self.session.preview(buffer_text)
        hit = self.session.labels[next_id] if next_id is not None else None
        shown = dice[:TOP]
        fragments = []
        for piece, p in shown:
            mark = "»" if piece == hit else " "
            style = "class:die-hit" if piece == hit else "class:die-row"
            fragments.append((style, f" {mark} {piece} {p:6.1%}"))
        if hit is not None and hit not in [piece for piece, _ in shown]:
            p = next(p for piece, p in dice if piece == hit)
            fragments.append(("class:die-hit", f" » {hit} {p:6.1%}"))
        if len(dice) > TOP:
            fragments.append(("class:more", f"   ... {len(dice) - TOP} more"))
        return fragments


class GhostSuggestions(AutoSuggest):
    def __init__(self, chat: GhostChat):
        self._chat = chat
        self._cache: dict[str, tuple[str, int | None]] = {}

    def _roll_for(self, context: str) -> tuple[str, int | None]:
        if context not in self._cache:
            ids, roll = self._chat.roll(context)
            first = ids[0] if ids else None
            for offset in range(len(roll) + 1):
                self._cache[context + roll[:offset]] = (roll[offset:], first)
        return self._cache[context]

    def real_for(self, context: str) -> str:
        return self._roll_for(context)[0]

    def first_id_for(self, context: str) -> int | None:
        return self._roll_for(context)[1]

    def get_suggestion(self, buffer, document):
        roll = self.real_for(document.text)
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
    return chat.die_bar(buffer.text, suggestions.first_id_for(buffer.text))


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

    editor_keys = KeyBindings()

    @editor_keys.add("tab")
    def _(event):
        if chat.session is None:
            return
        real = suggestions.real_for(buffer.text)
        if not real:
            return
        chat.accept_rest(buffer.text, real)
        buffer.text = chat.session.text
        buffer.cursor_position = len(buffer.text)

    @editor_keys.add("enter")
    def _(event):
        if chat.session is not None:
            chat.sync(buffer.text)

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
    layout = Layout(
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
