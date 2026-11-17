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
from prompt_toolkit.widgets import Frame, MenuItem, TextArea

from murrai.presentation.chat.form import FormOverlay, TrainingForm
from murrai.presentation.chat.lanes import (
    SILENT,
    CommandPicked,
    SessionSwapped,
)
from murrai.presentation.chat.menu import (
    AboutPopover,
    ClickableFormattedText,
    MenuBar,
)
from murrai.presentation.chat.panel import (
    CommandPanel,
    FormCommand,
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
ESC_FLUSH = 0.1
PANEL_WIDTH = 36
BREAK = "↵"
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
        "menu-bar": "bg:ansiblue fg:ansiwhite",
        "menu-bar.selected-item": "bg:ansiwhite fg:ansiblue",
        "menu": "bg:ansiblue fg:ansiwhite",
        "menu-border": "bg:ansiblue fg:ansiblue",
        "menu-item": "bg:ansiblue fg:ansiwhite",
        "menu-item.selected": "bg:ansiwhite fg:ansiblue",
        "shadow": "bg:#000087",
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
            shown_piece = piece.replace("\n", BREAK)
            fragments.append((style, f" {mark} {shown_piece} {p:6.1%}"))
        if hit is not None and hit not in [piece for piece, _ in shown]:
            p = next(p for piece, p in dice if piece == hit)
            marked = hit.replace("\n", BREAK)
            fragments.append(("class:die-hit", f" » {marked} {p:6.1%}"))
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
        return Suggestion(roll.replace("\n", BREAK))

    def forget(self) -> None:
        self._cache.clear()


def story_fragments(session, tail_lines=None):
    if session is None:
        return []
    lines = []
    for segment, origin in session.segments():
        for index, piece in enumerate(segment.split("\n")):
            lines.append((piece, origin, index > 0))
    if tail_lines is not None:
        lines = lines[-tail_lines:]
    fragments = []
    for index, (text, origin, continues) in enumerate(lines):
        if index and continues:
            fragments.append(("", "\n"))
        style = "class:generated" if origin == "generated" else ""
        fragments.append((style, text))
    return fragments


def _die_fragments(chat, suggestions, buffer):
    return chat.die_bar(buffer.text, suggestions.first_id_for(buffer.text))


def _story_tail(window):
    info = window.render_info
    if info is None:
        return STORY_TAIL_LINES
    return max(info.window_height, 1)


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


def chat_body(
    story,
    editor,
    die,
    panel_list,
    panel,
    status,
    chat,
    status_click,
):
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
    status_control = ClickableFormattedText(
        on_click=status_click,
        text=lambda: status_line(
            status,
            chat.session.kind if chat.session else None,
        ),
    )
    return HSplit(
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
                status_control,
                height=1,
                dont_extend_height=True,
            ),
        ],
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

    about = AboutPopover(on_close=lambda: focus(buffer), focus=focus)

    editor_keys = KeyBindings()

    @editor_keys.add("tab")
    def _(event):
        if chat.session is None:
            return
        real = suggestions.real_for(buffer.text)
        if not real:
            return
        chat.accept_rest(buffer.text, real)
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
            elif isinstance(command, FormCommand):
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

    def menu_focused():
        return app.layout.current_window == menu_bar.window

    @global_keys.add("f9")
    def _(event):
        if overlay.visible or about.visible:
            return
        if menu_focused():
            event.app.layout.focus(buffer)
        else:
            event.app.layout.focus(menu_bar.window)

    @global_keys.add("escape", filter=Condition(lambda: menu_focused()))
    def _(event):
        event.app.layout.focus(buffer)

    story = Window(
        ClickableFormattedText(
            on_click=lambda event: focus(buffer),
            text=lambda: (
                story_fragments(chat.session, tail_lines=_story_tail(story))
                if chat.session
                else [("class:hint", "no model yet; ctrl-p opens commands")]
            ),
        ),
        wrap_lines=True,
    )
    die = Window(
        ClickableFormattedText(
            on_click=lambda event: focus(buffer),
            text=lambda: _die_fragments(chat, suggestions, buffer),
        ),
        height=1,
    )
    panel_list = Window(
        ClickableFormattedText(
            on_click=lambda event: focus(panel_list),
            text=lambda: list_fragments(panel),
            focusable=True,
            key_bindings=panel_keys,
        ),
    )
    menu_bar = MenuBar(
        body=chat_body(
            story=story,
            editor=editor,
            die=die,
            panel_list=panel_list,
            panel=panel,
            status=status,
            chat=chat,
            status_click=lambda event: focus(buffer),
        ),
        menu_items=[
            MenuItem("Models"),
            MenuItem("Sessions"),
            MenuItem("About", children=[MenuItem("about", handler=about.open)]),
        ],
        floats=[overlay.float, about.float],
    )

    @menu_bar.control.key_bindings.add("c-c", eager=True)
    def _(event):
        event.app.exit()

    layout = Layout(menu_bar.container)
    editor.control.key_bindings = editor_keys
    app = Application(
        layout=layout,
        key_bindings=global_keys,
        style=STYLE,
        full_screen=True,
        cursor=CursorShape.BLOCK,
        mouse_support=True,
        input=input,
        output=output,
    )
    app.ttimeoutlen = ESC_FLUSH
    app.timeoutlen = ESC_FLUSH
    status.redraw = app.invalidate
    app.layout.focus(buffer)
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
