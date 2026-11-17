from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Window
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.widgets import MenuItem

from murrai.presentation.chat.menu import (
    ClickableFormattedText,
    MenuBar,
    WideShadow,
    menu_tree,
)
from murrai.presentation.chat.panel import Command, FormCommand
from murrai.presentation.chat.repl import STYLE


def mouse(kind, x=0, y=0):
    return MouseEvent(Point(x=x, y=y), kind, MouseButton.LEFT, frozenset())


def container():
    return MenuBar(
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


def geometry(shadow_container):
    return [
        (f.left, f.right, f.top, f.bottom, f.width, f.height)
        for f in shadow_container.floats
    ]


def test_every_dropdown_level_wears_the_wide_shadow():
    menubar = container()
    wanted = geometry(WideShadow(Window()).container)

    levels = menubar.container.floats[:3]

    assert [geometry(level.content.content) for level in levels] == [wanted] * 3


def _leaf(name, continues=False):
    return FormCommand(
        name,
        "knobs",
        lambda: None,
        fields=(),
        bake=lambda values: None,
        trains=True,
        continues=continues,
    )


def test_the_models_menu_lists_new_load_and_retrain():
    tree = {
        "new": (_leaf("counted bigram"), _leaf("neural bigram")),
        "load": (Command("load gpt-corpus", "saved yesterday", lambda: None),),
        "retrain": (_leaf("retrain gpt", continues=True),),
    }
    picked = []
    opened = []

    items = menu_tree(tree, pick=picked.append, about=lambda: opened.append(True))

    models = items[0]
    assert [item.text for item in models.children] == ["new", "load", "retrain"]
    assert [item.text for item in models.children[0].children] == [
        "counted bigram",
        "neural bigram",
    ]
    models.children[0].children[1].handler()
    assert picked[0].name == "neural bigram"

    load_item = models.children[1].children[0]
    assert load_item.text == "load gpt-corpus · saved yesterday"
    load_item.handler()
    assert picked[-1].name == "load gpt-corpus"

    retrain_item = models.children[2].children[0]
    retrain_item.handler()
    assert picked[-1].name == "retrain gpt"

    assert items[1].text == "Sessions"
    assert items[2].text == "About"
    items[2].children[0].handler()
    assert opened == [True]


def test_without_menu_content_the_bar_keeps_only_sessions_and_about():
    items = menu_tree(None, pick=lambda c: None, about=lambda: None)

    assert [item.text for item in items] == ["Models", "Sessions", "About"]


def test_the_submenu_opens_beside_the_parent_dropdown():
    menubar = MenuBar(
        body=Window(),
        menu_items=[MenuItem("Models", children=[MenuItem("gpt", handler=None)])],
        floats=[],
    )
    menubar.selected_menu = [0, 0, 0]

    fragments = menubar._submenu(0).content.text()

    lines = [[]]
    for fragment in fragments:
        if fragment[1] == "\n":
            lines.append([])
        else:
            lines[-1].append(fragment)
    lines = [line for line in lines if line]

    assert all(
        line[-1] == ("class:menu", " ")
        or line[-2:] == [("[SetMenuPosition]", ""), ("class:menu", " ")]
        for line in lines
    )
    marked = [
        line
        for line in lines
        if any(style == "[SetMenuPosition]" for style, *_ in line)
    ]
    assert len(marked) == 1
    assert marked[0][-2:] == [("[SetMenuPosition]", ""), ("class:menu", " ")]


def test_the_submenu_stays_closed_until_the_parent_is_entered():
    menubar = MenuBar(
        body=Window(),
        menu_items=[
            MenuItem("Models", children=[MenuItem("new", children=[MenuItem("gpt")])]),
        ],
        floats=[],
    )

    menubar.selected_menu = [0, 0]
    assert menubar._submenu(1).content.text() == []

    menubar.selected_menu = [0, 0, 0]
    assert menubar._submenu(1).content.text() != []


def test_the_submenu_floats_hide_the_closed_levels():
    menubar = MenuBar(
        body=Window(),
        menu_items=[
            MenuItem("Models", children=[MenuItem("new", children=[MenuItem("gpt")])]),
        ],
        floats=[],
    )

    assert menubar.container.floats[1].content.filter is not None
    assert menubar.container.floats[2].content.filter is not None


def test_clicking_an_item_with_children_enters_it():
    menubar = MenuBar(
        body=Window(),
        menu_items=[
            MenuItem("Models", children=[MenuItem("new", children=[MenuItem("gpt")])]),
        ],
        floats=[],
    )
    menubar.selected_menu = [0, 0]

    fragments = menubar._submenu(0).content.text()
    handler = next(f[2] for f in fragments if len(f) >= 3)
    handler(mouse(MouseEventType.MOUSE_UP))

    assert menubar.selected_menu == [0, 0, 0]


def test_the_dropdown_border_is_invisible_against_the_menu():
    border = STYLE.get_attrs_for_style_str("class:menu-border")

    assert border.bgcolor == STYLE.get_attrs_for_style_str("class:menu").bgcolor
    assert border.color == border.bgcolor


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


def test_hovering_moves_the_selection_and_clicks_wait_for_release():
    picked = []
    menubar = MenuBar(
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
