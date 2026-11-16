from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from murrai.presentation.chat.menu import (
    GAP,
    Menu,
    MenuBar,
    MenuItem,
    bar_fragments,
    description_fragments,
    dropdown_left,
    menu_columns,
    menu_fragments,
)

ABOUT_ITEM = MenuItem("about", "what murrAI is", lambda: None)
EVENT = MouseEvent(
    Point(x=0, y=0),
    MouseEventType.MOUSE_UP,
    MouseButton.LEFT,
    frozenset(),
)


def click(kind):
    return MouseEvent(
        Point(x=0, y=0),
        kind,
        MouseButton.LEFT,
        frozenset(),
    )


def bar():
    return MenuBar(
        (
            Menu("Models"),
            Menu("Sessions"),
            Menu("About", items=(ABOUT_ITEM,)),
        ),
    )


def _render_row(window, width=40):
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition

    screen = Screen(initial_width=width, initial_height=2)
    window.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(0, 0, width, 1),
        "",
        False,
        0,
    )
    return [screen.data_buffer[0][x].style for x in range(width)]


def test_the_bar_starts_closed():
    menubar = bar()

    assert not menubar.active
    assert menubar.open_index is None
    assert menubar.current is None


def test_opening_a_menu_selects_it_and_its_first_item():
    menubar = bar()

    menubar.open(2)

    assert menubar.active
    assert menubar.open_index == 2
    assert menubar.current is ABOUT_ITEM


def test_closing_clears_the_open_menu():
    menubar = bar()
    menubar.open(0)

    menubar.close()

    assert not menubar.active
    assert menubar.open_index is None


def test_moving_across_menus_keeps_the_dropdown_open_and_wraps():
    menubar = bar()
    menubar.open(2)

    menubar.move_menu(1)
    assert menubar.open_index == 0
    menubar.move_menu(1)
    assert menubar.open_index == 1
    menubar.move_menu(-1)
    assert menubar.open_index == 0
    menubar.move_menu(-1)
    assert menubar.open_index == 2


def test_up_and_down_move_within_the_open_menu_and_wrap():
    first = MenuItem("one", "first", lambda: None)
    second = MenuItem("two", "second", lambda: None)
    menubar = MenuBar((Menu("letters", items=(first, second)),))
    menubar.open(0)

    menubar.move_item(1)
    assert menubar.current is second
    menubar.move_item(1)
    assert menubar.current is first
    menubar.move_item(-1)
    assert menubar.current is second


def test_an_empty_menu_selects_nothing_and_picks_none():
    menubar = bar()
    menubar.open(0)

    menubar.move_item(1)

    assert menubar.current is None
    assert menubar.pick() is None
    assert not menubar.active


def test_pick_returns_the_selected_item_and_closes():
    menubar = MenuBar((Menu("letters", items=(ABOUT_ITEM,)),))
    menubar.open(0)

    picked = menubar.pick()

    assert picked is ABOUT_ITEM
    assert not menubar.active


def test_a_letter_picks_the_matching_item_of_the_open_menu():
    picked = []
    item = MenuItem("about", "d", lambda: picked.append(1))
    menubar = MenuBar((Menu("about", items=(item,)),))
    menubar.open(0)

    assert menubar.jump("a") is item
    assert not menubar.active


def test_a_letter_opens_the_menu_whose_title_starts_with_it():
    menubar = bar()

    assert menubar.jump("m") is None
    assert menubar.open_index == 0
    assert menubar.active

    menubar.close()
    assert menubar.jump("s") is None
    assert menubar.open_index == 1


def test_a_letter_falls_through_to_a_title_from_an_open_menu():
    menubar = bar()
    menubar.open(0)

    assert menubar.jump("a") is None
    assert menubar.open_index == 2


def test_a_letter_that_matches_nothing_changes_nothing():
    menubar = bar()
    menubar.open(0)

    assert menubar.jump("z") is None
    assert menubar.open_index == 0
    assert menubar.active


def test_the_bar_shows_titles_with_click_handlers_that_toggle():
    menubar = bar()

    fragments = bar_fragments(menubar)
    assert "".join(t for _, t, *rest in fragments) == "Models  Sessions  About"
    models = next(f for f in fragments if f[1] == "M")
    models[2](EVENT)
    assert menubar.open_index == 0
    models[2](EVENT)
    assert menubar.open_index is None


def test_the_leading_letter_is_capitalized_and_underlined():
    fragments = bar_fragments(bar())

    assert ("class:menu underline", "M", fragments[0][2]) in fragments
    assert ("class:menu", "odels", fragments[1][2]) in fragments
    assert ("class:menu underline", "S", fragments[3][2]) in fragments
    assert ("class:menu", "essions", fragments[4][2]) in fragments


def test_the_open_title_keeps_its_underline_while_selected():
    menubar = bar()
    menubar.open(1)

    fragments = bar_fragments(menubar)

    assert ("class:menu-selected underline", "S", fragments[3][2]) in fragments
    assert ("class:menu-selected", "essions", fragments[4][2]) in fragments


def test_clicking_another_title_switches_the_open_menu():
    menubar = bar()
    menubar.open(0)

    sessions = next(f for f in bar_fragments(menubar) if f[1] == "S")
    sessions[2](EVENT)

    assert menubar.open_index == 1


def test_the_open_title_is_styled_selected():
    menubar = bar()
    menubar.open(1)

    fragments = bar_fragments(menubar)

    titles = [f for f in fragments if f[1] != " " * GAP]
    assert titles[2][0] == "class:menu-selected underline"
    assert titles[3][0] == "class:menu-selected"
    assert titles[0][0] == "class:menu underline"


def test_the_closed_bar_styles_every_title_the_same():
    titles = [f for f in bar_fragments(bar()) if f[1] != " " * GAP]

    assert all(
        f[0].startswith("class:menu") and "menu-selected" not in f[0] for f in titles
    )


def test_the_dropdown_lists_the_open_menu_with_the_selected_item_marked():
    first = MenuItem("one", "first", lambda: None)
    second = MenuItem("two", "second", lambda: None)
    menubar = MenuBar((Menu("letters", items=(first, second)),))
    menubar.open(0)

    fragments = menu_fragments(menubar)

    one = next(f for f in fragments if f[1] == "one")
    two = next(f for f in fragments if f[1] == "two")
    assert one[0] == "class:menu-selected"
    assert two[0] == "class:menu"
    assert "".join(t for _, t, *_rest in fragments) == "\n  one  \n  two  \n\n"


def test_the_dropdown_margins_are_blank_lines_without_handlers():
    first = MenuItem("one", "first", lambda: None)
    second = MenuItem("two", "second", lambda: None)
    menubar = MenuBar((Menu("letters", items=(first, second)),))
    menubar.open(0)

    fragments = menu_fragments(menubar)

    assert fragments[0] == ("", "\n")
    assert fragments[-1] == ("", "\n")
    assert len(fragments[0]) == 2
    assert len(fragments[-1]) == 2
    assert fragments[4][0] == "class:menu"
    assert fragments[4][1] == "  "
    assert fragments[6][0] == "class:menu"
    assert fragments[6][1] == "  \n"


def test_the_selected_item_highlights_its_whole_row():
    first = MenuItem("one", "first", lambda: None)
    menubar = MenuBar((Menu("letters", items=(first,)),))
    menubar.open(0)

    fragments = menu_fragments(menubar)

    row = fragments[1:4]
    assert all(f[0] == "class:menu-selected" for f in row)


def test_the_dropdown_rows_carry_their_handler_from_edge_to_edge():
    picked = []
    item = MenuItem("one", "first", lambda: picked.append(1))
    menubar = MenuBar((Menu("letters", items=(item,)),))
    menubar.open(0)

    fragments = menu_fragments(menubar)

    assert all(len(f) == 3 for f in fragments[1:4])
    fragments[1][2](EVENT)
    assert picked == [1]


def test_the_dropdown_click_handler_picks_and_runs_the_action():
    picked = []
    item = MenuItem("about", "d", lambda: picked.append(1))
    menubar = MenuBar((Menu("about", items=(item,)),))
    menubar.open(0)

    menu_fragments(menubar)[2][2](EVENT)

    assert picked == [1]
    assert not menubar.active


def test_a_title_ignores_hover_scroll_and_press_and_clicks_on_release():
    menubar = bar()
    toggle = bar_fragments(menubar)[0][2]

    for kind in (
        MouseEventType.MOUSE_MOVE,
        MouseEventType.SCROLL_UP,
        MouseEventType.SCROLL_DOWN,
        MouseEventType.MOUSE_DOWN,
    ):
        toggle(click(kind))
        assert menubar.open_index is None, kind

    toggle(click(MouseEventType.MOUSE_UP))
    assert menubar.open_index == 0


def test_an_item_ignores_hover_scroll_and_press_and_picks_on_release():
    picked = []
    item = MenuItem("about", "d", lambda: picked.append(1))
    menubar = MenuBar((Menu("about", items=(item,)),))
    menubar.open(0)
    picker = next(f for f in menu_fragments(menubar) if f[1] == "about")[2]

    for kind in (
        MouseEventType.MOUSE_MOVE,
        MouseEventType.SCROLL_UP,
        MouseEventType.SCROLL_DOWN,
        MouseEventType.MOUSE_DOWN,
    ):
        picker(click(kind))
        assert menubar.active, kind
        assert picked == []

    picker(click(MouseEventType.MOUSE_UP))
    assert picked == [1]


def test_the_dropdown_is_empty_without_an_open_menu():
    assert menu_fragments(bar()) == []


def test_the_description_shows_the_selected_item():
    menubar = bar()
    menubar.open(2)

    assert "".join(t for _, t in description_fragments(menubar)) == "what murrAI is"

    menubar.open(0)
    assert description_fragments(menubar) == []


def test_the_description_is_empty_when_closed():
    assert description_fragments(bar()) == []


def test_the_dropdown_columns_follow_the_bar_titles():
    menubar = bar()

    assert menu_columns(menubar) == [0, 8, 18]


def test_the_dropdown_shifts_left_so_items_align_under_their_title():
    assert dropdown_left(bar(), 1) == 6
    assert dropdown_left(bar(), 2) == 16
    assert dropdown_left(bar(), 0) == 0


def test_the_menu_surfaces_hide_the_cursor():
    from prompt_toolkit.key_binding import KeyBindings

    from murrai.presentation.chat.repl import AboutDialog, menu_screen

    menu_bar_window, dropdowns = menu_screen(bar(), KeyBindings())
    assert menu_bar_window.content.show_cursor is False
    assert dropdowns[0].content.content.content.show_cursor is False

    about = AboutDialog(on_close=lambda: None, focus=lambda control: None)
    assert about.cross.content.show_cursor is False
    assert about.text.content.show_cursor is False


def test_the_about_popover_centers_itself_without_a_position():
    from murrai.presentation.chat.repl import AboutDialog

    about = AboutDialog(on_close=lambda: None, focus=lambda control: None)

    assert about.float.left is None
    assert about.float.top is None


def test_the_about_cross_closes_the_popover():
    from murrai.presentation.chat.repl import AboutDialog

    closed = []
    about = AboutDialog(on_close=lambda: closed.append(1), focus=lambda c: None)
    about.open()

    cross = next(f for f in about.cross_fragments() if f[1].strip() == "×")  # noqa: RUF001
    cross[2](EVENT)

    assert not about.visible
    assert closed == [1]


def test_the_bar_paints_its_whole_line_not_just_the_words():
    from prompt_toolkit.key_binding import KeyBindings

    from murrai.presentation.chat.repl import menu_screen

    menu_bar_window, _ = menu_screen(bar(), KeyBindings())

    styles = _render_row(menu_bar_window)

    assert all("class:menu" in style for style in styles)


def test_the_dropdown_paints_its_whole_box_not_just_the_words():
    from prompt_toolkit.key_binding import KeyBindings

    from murrai.presentation.chat.repl import menu_screen

    menubar = bar()
    menubar.open(2)
    _, dropdowns = menu_screen(menubar, KeyBindings())

    styles = _render_row(dropdowns[2].content, width=20)

    assert all("class:menu" in style for style in styles)
