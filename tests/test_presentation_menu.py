from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Window
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.widgets import MenuItem

from murrai.presentation.chat.repl import STYLE, ChatMenu


def mouse(kind, x=0, y=0):
    return MouseEvent(Point(x=x, y=y), kind, MouseButton.LEFT, frozenset())


def container():
    return ChatMenu(
        body=Window(),
        menu_items=[
            MenuItem(
                "About",
                children=[MenuItem("about"), MenuItem("longer item")],
            ),
        ],
    )


def dropdown_fragments(container, selected=None):
    container.selected_menu = selected or [0, 0]
    window = container._submenu(0)
    return window.content.text()


def test_the_dropdown_border_matches_the_shadow():
    attrs = STYLE.get_attrs_for_style_str("class:menu-border")

    assert attrs.bgcolor == STYLE.get_attrs_for_style_str("class:shadow").bgcolor


def test_the_dropdown_rows_are_left_aligned_with_a_two_space_pad():
    fragments = dropdown_fragments(container())

    first_row = next(f for f in fragments if "about" in f[1])

    assert first_row[1] == "  about".ljust(13) + " "


def test_the_longest_item_sets_the_dropdown_width():
    fragments = dropdown_fragments(container())

    second_row = next(f for f in fragments if "longer item" in f[1])

    assert second_row[1] == "  longer item "


def test_the_selected_row_is_marked_and_the_mark_is_invisible():
    fragments = dropdown_fragments(container())

    selected_row = next(f for f in fragments if "about" in f[1])

    assert selected_row[0] == "class:menu-item.selected"
    assert STYLE.get_attrs_for_style_str("class:menu-border").color == (
        STYLE.get_attrs_for_style_str("class:menu-border").bgcolor
    )


def test_hovering_moves_the_selection_and_clicks_wait_for_release():
    picked = []
    menubar = ChatMenu(
        body=Window(),
        menu_items=[
            MenuItem(
                "About",
                children=[
                    MenuItem("about", handler=lambda: picked.append(1)),
                ],
            ),
        ],
    )
    container.selected_menu = [0, 0]
    window = menubar._submenu(0)
    row = next(f for f in window.content.text() if "about" in f[1])

    row[2](mouse(MouseEventType.MOUSE_MOVE))
    assert menubar.selected_menu == [0, 0] or menubar.selected_menu == [0, 0]

    row[2](mouse(MouseEventType.MOUSE_DOWN))
    assert picked == []

    row[2](mouse(MouseEventType.MOUSE_UP))
    assert picked == [1]


def test_the_click_fallback_yields_to_handled_clicks():
    from murrai.presentation.chat.repl import ClickableFormattedText

    handled = []
    fallback = []
    control = ClickableFormattedText(
        on_click=lambda event: fallback.append(1),
        text=lambda: [("class:x", "cell", lambda event: handled.append(1))],
    )
    control.create_content(width=80, height=1)

    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, x=2))
    assert handled == [1]
    assert fallback == []

    control.mouse_handler(mouse(MouseEventType.MOUSE_UP, x=20))
    assert fallback == [1]
