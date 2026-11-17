from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window, to_container
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from murrai.presentation.chat.menu import ABOUT_TEXT, AboutPopover, WideShadow


def mouse(kind):
    return MouseEvent(Point(x=0, y=0), kind, MouseButton.LEFT, frozenset())


def about(closed):
    return AboutPopover(
        on_close=lambda: closed.append(1),
        focus=lambda control: None,
    )


def test_the_about_popover_centers_itself_without_a_position():
    dialog = about([])

    assert dialog.float.left is None
    assert dialog.float.top is None


def test_the_popover_is_exactly_as_wide_as_its_text_and_its_frame():
    dialog = about([])
    dialog.open()

    width = dialog.container.preferred_width(100).preferred

    assert width == len(ABOUT_TEXT) + 2


def painted_shadow(container, box_width, box_height):
    screen = Screen(initial_width=box_width + 6, initial_height=box_height + 4)
    container.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(0, 0, box_width, box_height),
        "",
        False,
        0,
    )
    screen.draw_all_floats()
    return {
        (x, y)
        for y in range(box_height + 4)
        for x in range(box_width + 6)
        if "shadow" in screen.data_buffer[y][x].style
    }


def test_the_shadow_is_two_cells_wide_on_the_right_and_one_row_deep():
    cells = painted_shadow(to_container(WideShadow(Window())), 10, 3)

    right = {(x, y) for x in (10, 11) for y in (1, 2, 3)}
    bottom = {(x, 3) for x in range(2, 12)}
    assert cells == right | bottom


def test_the_about_popover_never_shows_the_cursor():
    dialog = about([])

    assert dialog.cross.content.show_cursor is False
    assert dialog.text.content.show_cursor is False


def test_the_cross_closes_on_release_not_on_hover():
    closed = []
    dialog = about(closed)
    dialog.open()

    cross = next(f for f in dialog.cross_fragments() if f[1].strip() == "×")  # noqa: RUF001
    cross[2](mouse(MouseEventType.MOUSE_MOVE))
    cross[2](mouse(MouseEventType.MOUSE_DOWN))
    assert dialog.visible

    cross[2](mouse(MouseEventType.MOUSE_UP))
    assert not dialog.visible
    assert closed == [1]


def test_the_cross_sits_on_the_right_edge_of_the_popover():
    dialog = about([])
    dialog.open()

    screen = Screen(initial_width=30, initial_height=10)
    dialog.container.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(0, 0, 30, 8),
        "",
        False,
        0,
    )
    places = [
        (x, y)
        for y in range(10)
        for x in range(30)
        if screen.data_buffer[y][x].char == "×"  # noqa: RUF001
    ]

    assert places
    assert all(x > 20 for x, _ in places)
