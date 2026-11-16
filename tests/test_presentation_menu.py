from murrai.presentation.chat.menu import (
    GAP,
    Menu,
    MenuBar,
    MenuItem,
    bar_fragments,
    description_fragments,
    menu_columns,
    menu_fragments,
)

ABOUT_ITEM = MenuItem("about", "what murrAI is", lambda: None)
EVENT = object()


def bar():
    return MenuBar(
        (
            Menu("models"),
            Menu("sessions"),
            Menu("about", items=(ABOUT_ITEM,)),
        ),
    )


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
    assert "".join(t for _, t, *rest in fragments) == "models  sessions  about"
    models = next(f for f in fragments if f[1] == "models")
    models[2](EVENT)
    assert menubar.open_index == 0
    models[2](EVENT)
    assert menubar.open_index is None


def test_clicking_another_title_switches_the_open_menu():
    menubar = bar()
    menubar.open(0)

    sessions = next(f for f in bar_fragments(menubar) if f[1] == "sessions")
    sessions[2](EVENT)

    assert menubar.open_index == 1


def test_the_open_title_is_styled_selected():
    menubar = bar()
    menubar.open(1)

    fragments = bar_fragments(menubar)

    titles = [f for f in fragments if f[1] != " " * GAP]
    assert ("class:menu-selected", "sessions", titles[1][2]) in titles
    assert ("class:menu", "models", titles[0][2]) in titles


def test_the_closed_bar_styles_every_title_the_same():
    titles = [f for f in bar_fragments(bar()) if f[1] != " " * GAP]

    assert all(f[0] == "class:menu" for f in titles)


def test_the_dropdown_lists_the_open_menu_with_the_selected_item_marked():
    first = MenuItem("one", "first", lambda: None)
    second = MenuItem("two", "second", lambda: None)
    menubar = MenuBar((Menu("letters", items=(first, second)),))
    menubar.open(0)

    fragments = menu_fragments(menubar)

    assert ("class:menu-selected", "one", fragments[0][2]) in fragments
    assert ("class:menu", "two", fragments[2][2]) in fragments
    assert "".join(t for _, t, *rest in fragments) == "one\ntwo\n"


def test_the_dropdown_click_handler_picks_and_runs_the_action():
    picked = []
    item = MenuItem("about", "d", lambda: picked.append(1))
    menubar = MenuBar((Menu("about", items=(item,)),))
    menubar.open(0)

    menu_fragments(menubar)[0][2](EVENT)

    assert picked == [1]
    assert not menubar.active


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
