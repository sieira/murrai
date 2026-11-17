from importlib import metadata

from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    FloatContainer,
    WindowAlign,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import Frame, MenuContainer, MenuItem
from prompt_toolkit.widgets.base import Border


class ClickableFormattedText(FormattedTextControl):
    def __init__(self, on_click, **kwargs):
        super().__init__(**kwargs)
        self._on_click = on_click

    def mouse_handler(self, mouse_event):
        handled = super().mouse_handler(mouse_event)
        if (
            handled is NotImplemented
            and mouse_event.event_type == MouseEventType.MOUSE_UP
        ):
            self._on_click(mouse_event)
        return handled


ABOUT_TEXT = f"""\
murrAI {metadata.version("murrai")}

a conversational language model,
built from scratch on one machine

it aims to match or beat GPT-2-small
per parameter (ADR-0004)

GPL-3.0-only
"""


class MenuBar(MenuContainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for level in self.container.floats[:3]:
            level.content.content = WideShadow(
                level.content.content.content,
            ).container
        bar = self.window

        @Condition
        def has_focus():
            return get_app().layout.current_window == bar

        for level, float_ in enumerate(self.container.floats[:3]):
            if level:
                float_.content.filter = has_focus & Condition(
                    lambda level=level: len(self.selected_menu) > level + 1,
                )

        @self.control.key_bindings.add("space")
        def _(event):
            item = self._get_menu(len(self.selected_menu) - 1)
            if item.handler:
                event.app.layout.focus_last()
                item.handler()

    def _submenu(self, level=0):
        def get_text_fragments():
            result = []
            entered = level == 0 or level + 1 < len(self.selected_menu)
            if level < len(self.selected_menu) and entered:
                menu = self._get_menu(level)
                if menu.children:
                    width = menu.width + 3
                    result.append(("class:menu-border", Border.TOP_LEFT))
                    result.append(("class:menu-border", Border.HORIZONTAL * width))
                    result.append(("class:menu-border", Border.TOP_RIGHT))
                    result.append(("class:menu", " "))
                    result.append(("", "\n"))
                    selected_item = (
                        self.selected_menu[level + 1]
                        if level + 1 < len(self.selected_menu)
                        else None
                    )

                    def one_item(i, item):
                        def mouse_handler(mouse_event):
                            if item.disabled:
                                return
                            hover = mouse_event.event_type == MouseEventType.MOUSE_MOVE
                            if hover:
                                self.selected_menu = [
                                    *self.selected_menu[: level + 1],
                                    i,
                                ]
                            elif (
                                mouse_event.event_type == MouseEventType.MOUSE_UP
                                and item.handler
                            ):
                                get_app().layout.focus_last()
                                item.handler()
                            elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                                self.selected_menu = [
                                    *self.selected_menu[: level + 1],
                                    i,
                                    0,
                                ]

                        if i == selected_item:
                            style = "class:menu-item.selected"
                        else:
                            style = "class:menu-item"
                        yield ("class:menu-border", Border.VERTICAL)
                        yield (
                            style,
                            f"  {item.text}".ljust(width - 1)
                            + (">" if item.children else " "),
                            mouse_handler,
                        )
                        yield ("class:menu-border", Border.VERTICAL)
                        if i == selected_item:
                            yield ("[SetMenuPosition]", "")
                            yield ("class:menu", " ")
                        yield ("", "\n")

                    for i, item in enumerate(menu.children):
                        result.extend(one_item(i, item))
                    result.append(("class:menu-border", Border.BOTTOM_LEFT))
                    result.append(("class:menu-border", Border.HORIZONTAL * width))
                    result.append(("class:menu-border", Border.BOTTOM_RIGHT))
                    result.append(("class:menu", " "))
            return result

        return Window(
            FormattedTextControl(get_text_fragments, show_cursor=False),
            style="class:menu",
        )


class WideShadow:
    def __init__(self, body):
        self.container = FloatContainer(
            content=body,
            floats=[
                Float(
                    bottom=-1,
                    height=1,
                    left=2,
                    right=-2,
                    transparent=True,
                    content=Window(style="class:shadow"),
                ),
                Float(
                    bottom=-1,
                    top=1,
                    width=2,
                    right=-2,
                    transparent=True,
                    content=Window(style="class:shadow"),
                ),
            ],
        )

    def __pt_container__(self):
        return self.container


class AboutPopover:
    def __init__(self, on_close, focus):
        self.visible = False
        self._on_close = on_close
        self._focus = focus
        keys = KeyBindings()

        @keys.add("escape")
        @keys.add("enter")
        def _(event):
            self.close()

        self.cross = Window(
            FormattedTextControl(
                lambda: self.cross_fragments(),
                focusable=True,
                key_bindings=keys,
                show_cursor=False,
            ),
            height=1,
            align=WindowAlign.RIGHT,
        )
        self.text = Window(
            FormattedTextControl(
                lambda: [("", ABOUT_TEXT)],
                show_cursor=False,
            ),
        )
        self.shadow = WideShadow(
            body=Frame(
                body=HSplit([self.cross, self.text]),
                title="about",
                style="class:dialog.body",
                modal=True,
            ),
        )
        self.container = ConditionalContainer(
            self.shadow,
            Condition(lambda: self.visible),
        )
        self.float = Float(content=self.container)

    def cross_fragments(self):
        def close(event):
            if event.event_type == MouseEventType.MOUSE_UP:
                self.close()

        return [("class:failed", " ×", close)]  # noqa: RUF001

    def open(self):
        self.visible = True
        self._focus(self.cross)

    def close(self):
        self.visible = False
        self._on_close()


def menu_tree(content, pick, about):
    if content is None:
        return [
            MenuItem("Models"),
            MenuItem("Sessions"),
            MenuItem("About", children=[MenuItem("about", handler=about)]),
        ]

    def leaf(item):
        return lambda: pick(item)

    def item(item):
        return MenuItem(f"{item.name} · {item.description}", handler=leaf(item))

    retrain_items = [item(entry) for entry in content["retrain"]]
    models = MenuItem(
        "Models",
        children=[
            MenuItem("new", children=[item(entry) for entry in content["new"]]),
            MenuItem("load", children=[item(entry) for entry in content["load"]]),
            MenuItem("retrain", children=retrain_items),
        ],
    )
    sessions = MenuItem(
        "Sessions",
        children=[item(entry) for entry in content["sessions"]],
    )
    return [
        models,
        sessions,
        MenuItem("About", children=[MenuItem("about", handler=about)]),
    ]
